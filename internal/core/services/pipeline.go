package services

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/leotulipan/transcribe/internal/adapters/audio"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// pipelineRun executes the full transcription pipeline. Returns the final
// Result + error; emits progress events to `emit` as it walks the stages.
func pipelineRun(ctx context.Context, req domain.Request, deps Deps, emit func(domain.ProgressEvent)) (result *domain.Result, err error) {
	prov, err := providerFor(deps, req.Provider)
	if err != nil {
		return nil, err
	}
	model := req.Model
	if model == "" {
		model = prov.DefaultModel()
	}
	caps := prov.Capabilities(model)

	// Stage 0 — capability check
	if err = checkCapabilities(req, model, caps); err != nil {
		return nil, err
	}

	// Stage 1 — probe
	emit(domain.ProgressEvent{Stage: domain.StageProbing})
	src, err := deps.Audio.Probe(req.InputPath)
	if err != nil {
		return nil, fmt.Errorf("probe: %w", err)
	}

	// Stage 2 — result-cache lookup
	var cached *domain.Result
	if req.UseCache && deps.Cache != nil {
		r, hit, lookupErr := deps.Cache.Lookup(req.InputPath, req.Provider)
		if lookupErr != nil {
			if deps.Log != nil {
				deps.Log.Warn("result cache lookup failed", "err", lookupErr)
			}
		} else if hit {
			cached = r
		}
	}

	var tempFiles []domain.AudioFile
	var workDir string

	// Deferred cleanup with policy from spec §6.3.1
	defer func() {
		keepFiles := false
		for _, tf := range tempFiles {
			if !tf.IsTemp {
				continue
			}
			switch {
			case !tf.Complete:
				_ = deps.Audio.Cleanup(tf)
			case req.KeepIntermediates:
				// User asked to retain all intermediates — skip cleanup.
				keepFiles = true
			case req.KeepFLACIntermediates && (tf.Codec == "flac" || tf.Container == "flac"):
				// User asked to retain FLAC intermediates specifically.
				keepFiles = true
			case err == nil:
				_ = deps.Audio.Cleanup(tf)
			case transient(err):
				keepFiles = true // keep for future retry
			default:
				keepFiles = true // keep (permanent error — don't delete intermediates)
			}
		}
		// Remove empty temp directories unless we kept files for retry.
		if !keepFiles && workDir != "" {
			cleanupEmptyWorkDir(workDir)
		}
	}()

	if cached == nil {
		// Stage 3 — working dir
		workDir, _ = resolveWorkDir(req.InputPath)

		// Stage 4 — intermediate cache
		var prepared domain.AudioFile
		srcMTime := int64(0)
		if info, statErr := os.Stat(req.InputPath); statErr == nil {
			srcMTime = info.ModTime().Unix()
		}
		targetCodec := preferredCodecFor(req.Provider, caps)
		if req.UsePCM {
			targetCodec = "pcm_s16le"
		}
		prepOpts := ports.PrepareOpts{
			UseInput:           req.UseInput,
			SizeThresholdBytes: req.SizeThresholdBytes,
		}
		if hit := lookupIntermediate(workDir, src, srcMTime, req.Provider, model, prov.MaxUploadBytes(), targetCodec); hit != nil {
			prepared = *hit
		} else {
			// Stage 5 — prepare
			emit(domain.ProgressEvent{Stage: domain.StageCompressing})
			p, perr := prepare(ctx, deps.Audio, src, caps, prov.MaxUploadBytes(), workDir, ports.TargetFormat{Codec: targetCodec}, prepOpts)
			if perr != nil {
				err = perr
				return nil, fmt.Errorf("prepare: %w", err)
			}
			prepared = p
			// Write meta sidecar so future runs find it
			if prepared.IsTemp {
				_ = audio.WriteMeta(prepared.Path, audio.MetaInfo{
					Operation:       "transcode-or-copy",
					SourcePath:      req.InputPath,
					SourceSize:      src.SizeBytes,
					SourceMTimeUnix: srcMTime,
					TargetCodec:     prepared.Codec,
					TargetContainer: prepared.Container,
					MaxBytesBudget:  prov.MaxUploadBytes(),
					Provider:        req.Provider,
					Model:           model,
				})
			}
		}
		if prepared.IsTemp {
			tempFiles = append(tempFiles, prepared)
		}

		// Stage 6 — chunk (single-chunk path is common)
		emit(domain.ProgressEvent{Stage: domain.StageChunking})
		chunkOpts := ports.ChunkOpts{
			ChunkLengthSec: req.ChunkLengthSec,
			OverlapSec:     req.OverlapSec,
		}
		chunks, cerr := deps.Audio.Chunk(ctx, prepared, prov.MaxUploadBytes(), workDir, chunkOpts)
		if cerr != nil {
			err = cerr
			return nil, fmt.Errorf("chunk: %w", err)
		}
		// Track chunk files for cleanup when chunking actually split the source.
		// audio.Chunk returns a single chunk pointing at prepared.Path when no
		// split was needed — that file is already in tempFiles via `prepared`.
		if len(chunks) > 1 || (len(chunks) == 1 && chunks[0].Path != prepared.Path) {
			for _, c := range chunks {
				tempFiles = append(tempFiles, domain.AudioFile{
					Path:     c.Path,
					IsTemp:   true,
					Complete: c.Complete,
				})
			}
		}

		// Stage 7 — transcribe each chunk
		emit(domain.ProgressEvent{Stage: domain.StageTranscribing})
		var parts []*domain.Result
		for i, c := range chunks {
			emit(domain.ProgressEvent{
				Stage:   domain.StageTranscribing,
				Percent: float64(i) / float64(len(chunks)),
				Message: fmt.Sprintf("chunk %d/%d", i+1, len(chunks)),
			})
			chunkAudio := domain.AudioFile{
				Path: c.Path, SizeBytes: c.SizeBytes, Codec: prepared.Codec, Container: prepared.Container,
				IsTemp: prepared.IsTemp, Complete: c.Complete,
			}
			r, terr := prov.Transcribe(ctx, chunkAudio, ports.ProviderOpts{
					Model:         model,
					Language:      req.Language,
					SpeakerLabels: req.SpeakerLabels,
					NumSpeakers:   req.NumSpeakers,
					KeyTerms:      req.KeyTerms,
					SpeechModels:  req.SpeechModels,
				})
			if terr != nil {
				err = terr
				return nil, terr
			}
			parts = append(parts, r)
		}
		merged, merr := mergeChunks(parts, chunks)
		if merr != nil {
			err = merr
			return nil, merr
		}
		merged.SourcePath = req.InputPath
		merged.Provider = req.Provider
		merged.Model = model
		result = merged
	} else {
		result = cached
	}

	// DaVinci post-processing (only mutates words; deterministic)
	if hasFormat(req.Formats, domain.FormatDavinciSRT) {
		applyDavinci(result, req.DaVinciOpts)
	}

	// Stage 8 — cache write (only when we actually transcribed)
	if cached == nil && deps.Cache != nil {
		if werr := deps.Cache.Save(req.InputPath, result); werr != nil && deps.Log != nil {
			deps.Log.Warn("cache save failed", "err", werr)
		}
	}

	// Stage 9 — write outputs
	emit(domain.ProgressEvent{Stage: domain.StageWriting})
	writeOpts := domain.WriteOpts{
		MaxCharsPerLine:  req.MaxCharsPerLine,
		SpeakerLabels:    req.SpeakerLabels,
		WordsPerSubtitle: req.WordsPerSubtitle,
		StartHour:        req.StartHour,
	}
	for i, f := range req.Formats {
		w, ok := deps.Writers[f]
		if !ok {
			err = fmt.Errorf("no writer registered for format %q", f)
			return nil, err
		}
		dst := outputPath(req, f)
		if werr := w.Write(result, dst, writeOpts); werr != nil {
			err = werr
			return nil, werr
		}
		emit(domain.ProgressEvent{Stage: domain.StageWriting, Percent: float64(i+1) / float64(len(req.Formats))})
	}

	emit(domain.ProgressEvent{Stage: domain.StageDone})
	return result, nil
}

func checkCapabilities(req domain.Request, model string, caps ports.ModelCapabilities) error {
	for _, f := range req.Formats {
		if f.NeedsTimestamps() && !caps.WordTimestamps {
			return domain.ErrIncompatible{
				Provider: req.Provider, Model: model, Format: f,
				Reason: "model does not return word-level timestamps",
			}
		}
	}
	return nil
}

func hasFormat(formats []domain.OutputFormat, f domain.OutputFormat) bool {
	for _, x := range formats {
		if x == f {
			return true
		}
	}
	return false
}

// preferredCodecFor returns the preferred transcode target codec for a provider.
// Constrained to a codec that appears in caps.AcceptedInputs so the prepared
// file is actually accepted by the model.
func preferredCodecFor(p domain.ProviderID, caps ports.ModelCapabilities) string {
	pref := []string{}
	switch p {
	case domain.ProviderAssemblyAI, domain.ProviderElevenLabs:
		pref = []string{"flac", "mp3"}
	default:
		pref = []string{"mp3", "flac"}
	}
	for _, codec := range pref {
		if codecOnlyAccepted(caps, codec) {
			return codec
		}
	}
	return "mp3" // fallback; transcode will surface an error if rejected
}

// resolveWorkDir picks the per-job temp directory next to the source file,
// falling back to os.TempDir() if the source-adjacent path isn't writable.
func resolveWorkDir(inputPath string) (string, bool) {
	base := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))
	sideBySide := filepath.Join(filepath.Dir(inputPath), ".transcribe-tmp", base)
	if err := os.MkdirAll(sideBySide, 0o755); err == nil {
		return sideBySide, true
	} else if !errors.Is(err, fs.ErrPermission) {
		return sideBySide, true
	}
	fallback := filepath.Join(os.TempDir(), "transcribe-"+base)
	_ = os.MkdirAll(fallback, 0o755)
	return fallback, false
}

// cleanupEmptyWorkDir removes workDir if it is empty, then removes its parent
// if the parent is named ".transcribe-tmp" and also becomes empty. Tolerates
// "not exist" and "not empty" silently. Other errors are also swallowed —
// callers do not need to act on cleanup failures.
func cleanupEmptyWorkDir(workDir string) {
	if workDir == "" {
		return
	}
	if err := os.Remove(workDir); err != nil {
		// ErrNotExist and "directory not empty" are both expected — ignore.
		return
	}
	// Job dir was removed; try the parent only when it is our own staging dir.
	parent := filepath.Dir(workDir)
	if filepath.Base(parent) != ".transcribe-tmp" {
		return
	}
	_ = os.Remove(parent) // silently ignore: non-empty parent is normal
}

// outputPath returns the path for an output file. If req.OutputDir is set the
// output lands there with the source basename; otherwise it lands next to the
// source.
func outputPath(req domain.Request, f domain.OutputFormat) string {
	base := strings.TrimSuffix(filepath.Base(req.InputPath), filepath.Ext(req.InputPath))
	dir := req.OutputDir
	if dir == "" {
		dir = filepath.Dir(req.InputPath)
	}
	var ext string
	switch f {
	case domain.FormatText:
		ext = ".txt"
	case domain.FormatSRT:
		ext = ".srt"
	case domain.FormatDavinciSRT:
		ext = ".davinci.srt"
	default:
		ext = "." + string(f)
	}
	return filepath.Join(dir, base+ext)
}
