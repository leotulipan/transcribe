package audio

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// Chunk slices `in` into chunks each ≤ maxBytes. Uses stream-copy so the chunk
// retains the source codec. opts.ChunkLengthSec overrides the bitrate-derived
// duration; opts.OverlapSec shifts each chunk's start earlier so consecutive
// chunks share overlapping audio (words in the overlap are double-transcribed —
// mergeChunks does not deduplicate them).
func (f *FFmpeg) Chunk(ctx context.Context, in domain.AudioFile, maxBytes int64, workDir string, opts ports.ChunkOpts) ([]domain.Chunk, error) {
	if in.SizeBytes <= maxBytes && opts.ChunkLengthSec == 0 {
		return []domain.Chunk{{Path: in.Path, StartOffset: 0, SizeBytes: in.SizeBytes, Complete: true}}, nil
	}
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		return nil, err
	}

	if in.Duration <= 0 {
		return nil, fmt.Errorf("chunk: source duration unknown")
	}

	var chunkDur time.Duration
	if opts.ChunkLengthSec > 0 {
		// User-supplied chunk length overrides the bitrate-derived budget.
		chunkDur = time.Duration(opts.ChunkLengthSec) * time.Second
	} else {
		// Derive from byte budget: bytes-per-second estimate with 90% margin.
		bps := float64(in.SizeBytes) / in.Duration.Seconds()
		chunkSec := (float64(maxBytes) * 0.9) / bps
		// Floor: at least 0.5 s (to avoid degenerate tiny chunks) but no more than
		// half the total duration so callers can always get at least 2 chunks when
		// the budget is less than half the file size.
		minSec := 0.5
		if minSec > in.Duration.Seconds()/2 {
			minSec = in.Duration.Seconds() / 2
		}
		if chunkSec < minSec {
			chunkSec = minSec
		}
		chunkDur = time.Duration(chunkSec * float64(time.Second))
	}

	overlapDur := time.Duration(opts.OverlapSec) * time.Second

	base := strings.TrimSuffix(filepath.Base(in.Path), filepath.Ext(in.Path))
	ext := filepath.Ext(in.Path)

	// Determine format for ffmpeg -f flag from container name
	container := strings.TrimPrefix(ext, ".")
	fmtFlag := ""
	if fmtName, ok := containerFormat[container]; ok {
		fmtFlag = fmtName
	}

	var chunks []domain.Chunk
	// nominalOffset is the "clean" boundary advancing by chunkDur each iteration.
	// startOffset is what ffmpeg actually seeks to (shifted back by overlap).
	var nominalOffset time.Duration
	idx := 0
	for nominalOffset < in.Duration {
		idx++
		final := filepath.Join(workDir, fmt.Sprintf("%s-chunk%02d%s", base, idx, ext))
		partial := partialPath(final)

		// Apply overlap: start earlier than the nominal boundary (but not before 0).
		// Overlap extends backward into the previous chunk only; chunk duration stays chunkDur.
		startOffset := nominalOffset - overlapDur
		if startOffset < 0 {
			startOffset = 0
		}

		args := []string{
			"-y",
			"-ss", fmt.Sprintf("%.3f", startOffset.Seconds()),
			"-t", fmt.Sprintf("%.3f", chunkDur.Seconds()),
			"-i", in.Path,
			"-c", "copy",
		}
		if fmtFlag != "" {
			args = append(args, "-f", fmtFlag)
		}
		args = append(args, partial)

		cmd := exec.CommandContext(ctx, f.ffmpeg, args...)
		hideConsole(cmd)
		if out, err := cmd.CombinedOutput(); err != nil {
			_ = os.Remove(partial)
			return nil, fmt.Errorf("ffmpeg chunk: %w: %s", err, string(out))
		}
		size, err := promote(final)
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, domain.Chunk{
			Path:        final,
			StartOffset: startOffset,
			SizeBytes:   size,
			Complete:    true,
		})
		nominalOffset += chunkDur
	}
	return chunks, nil
}
