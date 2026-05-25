package cache

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

const schemaVersion = 1

type envelope struct {
	SchemaVersion int               `json:"schema_version"`
	Provider      domain.ProviderID `json:"provider"`
	Model         string            `json:"model"`
	Language      string            `json:"language"`
	DurationMs    int64             `json:"duration_ms"`
	Text          string            `json:"text"`
	Confidence    float64           `json:"confidence"`
	Words         []wordJSON        `json:"words"`
	Segments      []segmentJSON     `json:"segments"`
	SourcePath    string            `json:"source_path"`
	Raw           json.RawMessage   `json:"raw,omitempty"`
}

type wordJSON struct {
	Text       string  `json:"text"`
	StartMs    int64   `json:"start_ms"`
	EndMs      int64   `json:"end_ms"`
	Confidence float64 `json:"confidence,omitempty"`
}

type segmentJSON struct {
	Text      string `json:"text"`
	StartMs   int64  `json:"start_ms"`
	EndMs     int64  `json:"end_ms"`
	SpeakerID string `json:"speaker_id,omitempty"`
}

type Sidecar struct{}

func New() *Sidecar { return &Sidecar{} }

// osWriteFile lives at package scope so tests can use it via the test helper.
var osWriteFile = func(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// sidecarPath returns the on-disk path of the cache file for (input, provider).
func sidecarPath(inputPath string, p domain.ProviderID) string {
	base := strings.TrimSuffix(inputPath, filepath.Ext(inputPath))
	return base + ".transcribe." + string(p) + ".json"
}

func (s *Sidecar) Lookup(inputPath string, p domain.ProviderID) (*domain.Result, bool, error) {
	data, err := os.ReadFile(sidecarPath(inputPath, p))
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return nil, false, nil
	case err != nil:
		return nil, false, err
	}
	var env envelope
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, false, err
	}
	if env.SchemaVersion != schemaVersion {
		return nil, false, nil
	}
	res := &domain.Result{
		Provider:   env.Provider,
		Model:      env.Model,
		Language:   env.Language,
		Text:       env.Text,
		Confidence: env.Confidence,
		Duration:   time.Duration(env.DurationMs) * time.Millisecond,
		SourcePath: env.SourcePath,
		RawJSON:    []byte(env.Raw),
	}
	for _, w := range env.Words {
		res.Words = append(res.Words, domain.Word{
			Text:       w.Text,
			Start:      time.Duration(w.StartMs) * time.Millisecond,
			End:        time.Duration(w.EndMs) * time.Millisecond,
			Confidence: w.Confidence,
		})
	}
	for _, sg := range env.Segments {
		res.Segments = append(res.Segments, domain.Segment{
			Text:      sg.Text,
			Start:     time.Duration(sg.StartMs) * time.Millisecond,
			End:       time.Duration(sg.EndMs) * time.Millisecond,
			SpeakerID: sg.SpeakerID,
		})
	}
	return res, true, nil
}

// LoadFromFile reads the sidecar JSON at exactly jsonPath (no path construction)
// and returns the parsed Result. Returns an error if the file is missing, unreadable,
// or has an unrecognised schema version.
func (s *Sidecar) LoadFromFile(jsonPath string) (*domain.Result, error) {
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, err
	}
	var env envelope
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, err
	}
	if env.SchemaVersion != schemaVersion {
		return nil, fmt.Errorf("unsupported sidecar schema version %d (expected %d)", env.SchemaVersion, schemaVersion)
	}
	res := &domain.Result{
		Provider:   env.Provider,
		Model:      env.Model,
		Language:   env.Language,
		Text:       env.Text,
		Confidence: env.Confidence,
		Duration:   time.Duration(env.DurationMs) * time.Millisecond,
		SourcePath: env.SourcePath,
		RawJSON:    []byte(env.Raw),
	}
	for _, w := range env.Words {
		res.Words = append(res.Words, domain.Word{
			Text:       w.Text,
			Start:      time.Duration(w.StartMs) * time.Millisecond,
			End:        time.Duration(w.EndMs) * time.Millisecond,
			Confidence: w.Confidence,
		})
	}
	for _, sg := range env.Segments {
		res.Segments = append(res.Segments, domain.Segment{
			Text:      sg.Text,
			Start:     time.Duration(sg.StartMs) * time.Millisecond,
			End:       time.Duration(sg.EndMs) * time.Millisecond,
			SpeakerID: sg.SpeakerID,
		})
	}
	return res, nil
}

func (s *Sidecar) Save(inputPath string, r *domain.Result) error {
	env := envelope{
		SchemaVersion: schemaVersion,
		Provider:      r.Provider,
		Model:         r.Model,
		Language:      r.Language,
		DurationMs:    r.Duration.Milliseconds(),
		Text:          r.Text,
		Confidence:    r.Confidence,
		SourcePath:    inputPath,
		Raw:           json.RawMessage(r.RawJSON),
	}
	for _, w := range r.Words {
		env.Words = append(env.Words, wordJSON{
			Text: w.Text, StartMs: w.Start.Milliseconds(), EndMs: w.End.Milliseconds(),
			Confidence: w.Confidence,
		})
	}
	for _, sg := range r.Segments {
		env.Segments = append(env.Segments, segmentJSON{
			Text: sg.Text, StartMs: sg.Start.Milliseconds(), EndMs: sg.End.Milliseconds(),
			SpeakerID: sg.SpeakerID,
		})
	}
	data, err := json.MarshalIndent(env, "", "  ")
	if err != nil {
		return err
	}
	return osWriteFile(sidecarPath(inputPath, r.Provider), data)
}
