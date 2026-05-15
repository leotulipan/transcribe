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
