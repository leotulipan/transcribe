package services

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

func TestPipeline_HappyPath_TextOnly(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "talk.mp3")
	require.NoError(t, os.WriteFile(inputPath, []byte("\xff\xfb"), 0o644))

	audio := &fakeAudio{
		probeOut: domain.AudioFile{Path: inputPath, Codec: "mp3", Container: "mp3", SizeBytes: 100, Duration: time.Second},
	}
	prov := &fakeProviderFull{
		id:        domain.ProviderGroq,
		models:    []string{"whisper-large-v3"},
		maxUpload: 1024,
		caps: map[string]ports.ModelCapabilities{
			"whisper-large-v3": {
				WordTimestamps: true,
				AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}},
			},
		},
		result: &domain.Result{
			Text:     "hello world",
			Language: "en",
			Words: []domain.Word{
				{Text: "hello", Start: 0, End: 500 * time.Millisecond},
				{Text: "world", Start: 600 * time.Millisecond, End: 1100 * time.Millisecond},
			},
			RawJSON: json.RawMessage(`{"k":"v"}`),
		},
	}
	cache := newFakeCache()
	textWriter := &recordingWriter{format: domain.FormatText}

	deps := Deps{
		Providers: map[domain.ProviderID]ports.Provider{domain.ProviderGroq: prov},
		Audio:     audio,
		Cache:     cache,
		Writers:   map[domain.OutputFormat]ports.FormatWriter{domain.FormatText: textWriter},
	}

	svc := New(deps)
	job, err := svc.Submit(context.Background(), domain.Request{
		InputPath: inputPath,
		Provider:  domain.ProviderGroq,
		Model:     "whisper-large-v3",
		Formats:   []domain.OutputFormat{domain.FormatText},
		UseCache:  true,
	})
	require.NoError(t, err)

	var events []domain.Stage
	for ev := range job.Progress() {
		events = append(events, ev.Stage)
	}
	res, err := job.Wait()
	require.NoError(t, err)
	require.Equal(t, "hello world", res.Text)
	require.Equal(t, 1, cache.saves)
	require.Len(t, textWriter.paths, 1)
	require.Contains(t, events, domain.StageProbing)
	require.Contains(t, events, domain.StageTranscribing)
	require.Contains(t, events, domain.StageWriting)
	require.Contains(t, events, domain.StageDone)
}

func TestPipeline_RejectsIncompatibleFormat(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "talk.mp3")
	require.NoError(t, os.WriteFile(inputPath, []byte("\xff\xfb"), 0o644))

	prov := &fakeProviderFull{
		id:        domain.ProviderOpenAI,
		models:    []string{"gpt-4o-audio"},
		maxUpload: 1024,
		caps: map[string]ports.ModelCapabilities{
			"gpt-4o-audio": {WordTimestamps: false, AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}},
		},
	}
	svc := New(Deps{
		Providers: map[domain.ProviderID]ports.Provider{domain.ProviderOpenAI: prov},
		Audio:     &fakeAudio{},
		Cache:     newFakeCache(),
		Writers:   map[domain.OutputFormat]ports.FormatWriter{},
	})

	job, err := svc.Submit(context.Background(), domain.Request{
		InputPath: inputPath, Provider: domain.ProviderOpenAI, Model: "gpt-4o-audio",
		Formats: []domain.OutputFormat{domain.FormatSRT},
	})
	require.NoError(t, err)
	_, err = job.Wait()
	var ei domain.ErrIncompatible
	require.ErrorAs(t, err, &ei)
	require.Equal(t, domain.FormatSRT, ei.Format)
}

func TestPipeline_UsePCMOverridesPreferredCodec(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "talk.mp3")
	require.NoError(t, os.WriteFile(inputPath, []byte("\xff\xfb"), 0o644))

	var capturedTarget ports.TargetFormat
	audio := &fakeAudio{
		probeOut: domain.AudioFile{Path: inputPath, Codec: "mp3", Container: "mp3", SizeBytes: 100, Duration: time.Second},
	}
	// Override Transcode to capture the target format used.
	audio.transcOut = domain.AudioFile{Path: filepath.Join(dir, "out.wav"), Codec: "pcm_s16le", Container: "wav", IsTemp: true, Complete: true, SizeBytes: 50}

	prov := &fakeProviderFull{
		id:        domain.ProviderGroq,
		models:    []string{"whisper-large-v3"},
		maxUpload: 50, // force transcode: src SizeBytes 100 > 50 maxUpload
		caps: map[string]ports.ModelCapabilities{
			"whisper-large-v3": {
				WordTimestamps: false,
				AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}},
			},
		},
		result: &domain.Result{Text: "hello"},
	}

	// Capture the target by wrapping the Transcode call.
	transcribeFn := prov.transcribeFn
	_ = transcribeFn

	// Build a custom fakeAudio that records the target codec.
	recordingAudio := &recordingTranscodeAudio{
		fakeAudio: &fakeAudio{
			probeOut: domain.AudioFile{Path: inputPath, Codec: "mp3", Container: "mp3", SizeBytes: 100, Duration: time.Second},
			transcOut: domain.AudioFile{Path: filepath.Join(dir, "out.wav"), Codec: "pcm_s16le", Container: "wav", IsTemp: true, Complete: true, SizeBytes: 50},
		},
		capturedTarget: &capturedTarget,
	}

	textWriter := &recordingWriter{format: domain.FormatText}
	svc := New(Deps{
		Providers: map[domain.ProviderID]ports.Provider{domain.ProviderGroq: prov},
		Audio:     recordingAudio,
		Cache:     newFakeCache(),
		Writers:   map[domain.OutputFormat]ports.FormatWriter{domain.FormatText: textWriter},
	})

	job, err := svc.Submit(context.Background(), domain.Request{
		InputPath: inputPath,
		Provider:  domain.ProviderGroq,
		Model:     "whisper-large-v3",
		Formats:   []domain.OutputFormat{domain.FormatText},
		UsePCM:    true,
	})
	require.NoError(t, err)
	_, err = job.Wait()
	require.NoError(t, err)
	require.Equal(t, "pcm_s16le", capturedTarget.Codec, "UsePCM should override transcode target to pcm_s16le")
}

func TestPipeline_KeepIntermediatesPreventsCleanup(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "talk.mp3")
	require.NoError(t, os.WriteFile(inputPath, []byte("\xff\xfb"), 0o644))

	// Transcode produces a temp file; it should NOT be cleaned up.
	transcoded := filepath.Join(dir, "out.mp3")
	require.NoError(t, os.WriteFile(transcoded, []byte("x"), 0o644))

	audio := &fakeAudio{
		probeOut:  domain.AudioFile{Path: inputPath, Codec: "pcm_s16le", Container: "wav", SizeBytes: 100, Duration: time.Second},
		transcOut: domain.AudioFile{Path: transcoded, Codec: "mp3", Container: "mp3", IsTemp: true, Complete: true, SizeBytes: 50},
	}
	prov := &fakeProviderFull{
		id:        domain.ProviderGroq,
		models:    []string{"whisper-large-v3"},
		maxUpload: 200,
		caps: map[string]ports.ModelCapabilities{
			"whisper-large-v3": {WordTimestamps: false, AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}},
		},
		result: &domain.Result{Text: "hello"},
	}
	textWriter := &recordingWriter{format: domain.FormatText}
	svc := New(Deps{
		Providers: map[domain.ProviderID]ports.Provider{domain.ProviderGroq: prov},
		Audio:     audio,
		Cache:     newFakeCache(),
		Writers:   map[domain.OutputFormat]ports.FormatWriter{domain.FormatText: textWriter},
	})

	job, err := svc.Submit(context.Background(), domain.Request{
		InputPath:         inputPath,
		Provider:          domain.ProviderGroq,
		Model:             "whisper-large-v3",
		Formats:           []domain.OutputFormat{domain.FormatText},
		KeepIntermediates: true,
	})
	require.NoError(t, err)
	_, err = job.Wait()
	require.NoError(t, err)
	// KeepIntermediates means Cleanup should never have been called.
	require.Equal(t, 0, audio.cleanupCalls, "KeepIntermediates should prevent any cleanup calls")
}

func TestPipeline_KeepFLACOnlyPreservesFlacFiles(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "talk.mp3")
	require.NoError(t, os.WriteFile(inputPath, []byte("\xff\xfb"), 0o644))

	transcoded := filepath.Join(dir, "out.flac")
	require.NoError(t, os.WriteFile(transcoded, []byte("x"), 0o644))

	// Prepared file is FLAC — should be kept when KeepFLACIntermediates=true.
	audio := &fakeAudio{
		probeOut:  domain.AudioFile{Path: inputPath, Codec: "pcm_s16le", Container: "wav", SizeBytes: 100, Duration: time.Second},
		transcOut: domain.AudioFile{Path: transcoded, Codec: "flac", Container: "flac", IsTemp: true, Complete: true, SizeBytes: 50},
	}
	prov := &fakeProviderFull{
		id:        domain.ProviderGroq,
		models:    []string{"whisper-large-v3"},
		maxUpload: 200,
		caps: map[string]ports.ModelCapabilities{
			"whisper-large-v3": {WordTimestamps: false, AcceptedInputs: []domain.AudioFormat{{Codec: "flac"}}},
		},
		result: &domain.Result{Text: "hello"},
	}
	textWriter := &recordingWriter{format: domain.FormatText}
	svc := New(Deps{
		Providers: map[domain.ProviderID]ports.Provider{domain.ProviderGroq: prov},
		Audio:     audio,
		Cache:     newFakeCache(),
		Writers:   map[domain.OutputFormat]ports.FormatWriter{domain.FormatText: textWriter},
	})

	job, err := svc.Submit(context.Background(), domain.Request{
		InputPath:             inputPath,
		Provider:              domain.ProviderGroq,
		Model:                 "whisper-large-v3",
		Formats:               []domain.OutputFormat{domain.FormatText},
		KeepFLACIntermediates: true,
	})
	require.NoError(t, err)
	_, err = job.Wait()
	require.NoError(t, err)
	require.Equal(t, 0, audio.cleanupCalls, "KeepFLACIntermediates=true with flac output should prevent cleanup")
}

func TestPipeline_ResultCacheHit_SkipsAudioAndProvider(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "talk.mp3")
	require.NoError(t, os.WriteFile(inputPath, []byte("\xff\xfb"), 0o644))

	cached := &domain.Result{Text: "cached text", Provider: domain.ProviderGroq}
	cache := newFakeCache()
	require.NoError(t, cache.Save(inputPath, cached))

	audio := &fakeAudio{} // probeOut not set — pipeline still calls Probe but won't call transcode
	audio.probeOut = domain.AudioFile{Path: inputPath, Codec: "mp3", Container: "mp3", SizeBytes: 100, Duration: time.Second}

	prov := &fakeProviderFull{id: domain.ProviderGroq, models: []string{"whisper-large-v3"},
		maxUpload: 1024,
		caps: map[string]ports.ModelCapabilities{
			"whisper-large-v3": {WordTimestamps: true, AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}},
		},
		transcribeFn: func(context.Context, domain.AudioFile, ports.ProviderOpts) (*domain.Result, error) {
			t.Fatal("provider must not be called on cache hit")
			return nil, nil
		},
	}
	textWriter := &recordingWriter{format: domain.FormatText}

	svc := New(Deps{
		Providers: map[domain.ProviderID]ports.Provider{domain.ProviderGroq: prov},
		Audio:     audio,
		Cache:     cache,
		Writers:   map[domain.OutputFormat]ports.FormatWriter{domain.FormatText: textWriter},
	})
	job, err := svc.Submit(context.Background(), domain.Request{
		InputPath: inputPath, Provider: domain.ProviderGroq, Model: "whisper-large-v3",
		Formats: []domain.OutputFormat{domain.FormatText}, UseCache: true,
	})
	require.NoError(t, err)
	res, err := job.Wait()
	require.NoError(t, err)
	require.Equal(t, "cached text", res.Text)
	require.Len(t, textWriter.paths, 1)
}

func TestPipelineRun_UseJSONInputSkipsProvider(t *testing.T) {
	dir := t.TempDir()
	jsonPath := filepath.Join(dir, "talk.transcribe.groq.json")

	loaded := &domain.Result{Text: "from json", Provider: domain.ProviderGroq}
	cache := newFakeCache()
	cache.storeFile(jsonPath, loaded)

	textWriter := &recordingWriter{format: domain.FormatText}
	prov := &fakeProviderFull{
		id:        domain.ProviderGroq,
		models:    []string{"whisper-large-v3"},
		maxUpload: 1024,
		caps: map[string]ports.ModelCapabilities{
			"whisper-large-v3": {WordTimestamps: false, AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}},
		},
		transcribeFn: func(context.Context, domain.AudioFile, ports.ProviderOpts) (*domain.Result, error) {
			t.Fatal("provider must not be called when UseJSONInput=true")
			return nil, nil
		},
	}

	svc := New(Deps{
		Providers: map[domain.ProviderID]ports.Provider{domain.ProviderGroq: prov},
		Audio:     &fakeAudio{},
		Cache:     cache,
		Writers:   map[domain.OutputFormat]ports.FormatWriter{domain.FormatText: textWriter},
	})

	job, err := svc.Submit(context.Background(), domain.Request{
		InputPath:    jsonPath,
		Provider:     domain.ProviderGroq,
		Formats:      []domain.OutputFormat{domain.FormatText},
		UseJSONInput: true,
	})
	require.NoError(t, err)
	res, err := job.Wait()
	require.NoError(t, err)
	require.Equal(t, "from json", res.Text)
	require.Len(t, textWriter.paths, 1)
	// No save because UseCache=false (default) and SaveCleanedJSON=false (default).
	require.Equal(t, 0, cache.saves)
}

func TestPipelineRun_SaveCleanedJSONWritesEvenWhenCacheDisabled(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "talk.mp3")
	require.NoError(t, os.WriteFile(inputPath, []byte("\xff\xfb"), 0o644))

	audio := &fakeAudio{
		probeOut: domain.AudioFile{Path: inputPath, Codec: "mp3", Container: "mp3", SizeBytes: 100, Duration: time.Second},
	}
	prov := &fakeProviderFull{
		id:        domain.ProviderGroq,
		models:    []string{"whisper-large-v3"},
		maxUpload: 1024,
		caps: map[string]ports.ModelCapabilities{
			"whisper-large-v3": {WordTimestamps: false, AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}},
		},
		result: &domain.Result{Text: "saved"},
	}
	cache := newFakeCache()
	textWriter := &recordingWriter{format: domain.FormatText}

	svc := New(Deps{
		Providers: map[domain.ProviderID]ports.Provider{domain.ProviderGroq: prov},
		Audio:     audio,
		Cache:     cache,
		Writers:   map[domain.OutputFormat]ports.FormatWriter{domain.FormatText: textWriter},
	})

	job, err := svc.Submit(context.Background(), domain.Request{
		InputPath:       inputPath,
		Provider:        domain.ProviderGroq,
		Model:           "whisper-large-v3",
		Formats:         []domain.OutputFormat{domain.FormatText},
		UseCache:        false,
		SaveCleanedJSON: true,
	})
	require.NoError(t, err)
	_, err = job.Wait()
	require.NoError(t, err)
	require.Equal(t, 1, cache.saves, "SaveCleanedJSON=true must write JSON even when UseCache=false")
}

// ---------------------------------------------------------------------------
// outputPath unit tests
// ---------------------------------------------------------------------------

func TestOutputPath_StripsTranscribeSuffixWhenUseJSONInput(t *testing.T) {
	req := domain.Request{
		InputPath:    "/tmp/data/myfile.transcribe.groq.json",
		Provider:     domain.ProviderGroq,
		UseJSONInput: true,
	}
	got := outputPath(req, domain.FormatText)
	require.Equal(t, filepath.FromSlash("/tmp/data/myfile.txt"), got)
}

func TestOutputPath_StripsTranscribeSuffixWhenProviderInPathDiffers(t *testing.T) {
	// Provider in path is openai; req.Provider is groq. Stripping is filename-driven.
	req := domain.Request{
		InputPath:    "/tmp/data/myfile.transcribe.openai.json",
		Provider:     domain.ProviderGroq,
		UseJSONInput: true,
	}
	got := outputPath(req, domain.FormatSRT)
	require.Equal(t, filepath.FromSlash("/tmp/data/myfile.srt"), got)
}

func TestOutputPath_NoUseJSONInput_BehavesAsBefore(t *testing.T) {
	req := domain.Request{
		InputPath: "/tmp/data/myfile.mp3",
		Provider:  domain.ProviderGroq,
	}
	got := outputPath(req, domain.FormatText)
	require.Equal(t, filepath.FromSlash("/tmp/data/myfile.txt"), got)
}

func TestOutputPath_UseJSONInput_NonSidecarJSON(t *testing.T) {
	// Input is a plain .json (not a sidecar); don't strip anything magical.
	req := domain.Request{
		InputPath:    "/tmp/data/weird.json",
		Provider:     domain.ProviderGroq,
		UseJSONInput: true,
	}
	got := outputPath(req, domain.FormatText)
	require.Equal(t, filepath.FromSlash("/tmp/data/weird.txt"), got)
}

func TestOutputPath_UseJSONInput_DavinciSRT(t *testing.T) {
	req := domain.Request{
		InputPath:    "/tmp/data/myfile.transcribe.groq.json",
		Provider:     domain.ProviderGroq,
		UseJSONInput: true,
	}
	got := outputPath(req, domain.FormatDavinciSRT)
	require.Equal(t, filepath.FromSlash("/tmp/data/myfile.davinci.srt"), got)
}

func TestPipelineRun_NoSaveWhenBothFalse(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "talk.mp3")
	require.NoError(t, os.WriteFile(inputPath, []byte("\xff\xfb"), 0o644))

	audio := &fakeAudio{
		probeOut: domain.AudioFile{Path: inputPath, Codec: "mp3", Container: "mp3", SizeBytes: 100, Duration: time.Second},
	}
	prov := &fakeProviderFull{
		id:        domain.ProviderGroq,
		models:    []string{"whisper-large-v3"},
		maxUpload: 1024,
		caps: map[string]ports.ModelCapabilities{
			"whisper-large-v3": {WordTimestamps: false, AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}},
		},
		result: &domain.Result{Text: "no save"},
	}
	cache := newFakeCache()
	textWriter := &recordingWriter{format: domain.FormatText}

	svc := New(Deps{
		Providers: map[domain.ProviderID]ports.Provider{domain.ProviderGroq: prov},
		Audio:     audio,
		Cache:     cache,
		Writers:   map[domain.OutputFormat]ports.FormatWriter{domain.FormatText: textWriter},
	})

	job, err := svc.Submit(context.Background(), domain.Request{
		InputPath:       inputPath,
		Provider:        domain.ProviderGroq,
		Model:           "whisper-large-v3",
		Formats:         []domain.OutputFormat{domain.FormatText},
		UseCache:        false,
		SaveCleanedJSON: false,
	})
	require.NoError(t, err)
	_, err = job.Wait()
	require.NoError(t, err)
	require.Equal(t, 0, cache.saves, "UseCache=false + SaveCleanedJSON=false must not write JSON")
}
