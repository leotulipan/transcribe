package services

import (
	"context"
	"errors"
	"sync"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// fakeAudio records every call so tests can assert which path the prepare
// decision tree took.
type fakeAudio struct {
	mu         sync.Mutex
	probeOut   domain.AudioFile
	copyOut    domain.AudioFile
	extractOut domain.AudioFile
	transcOut  domain.AudioFile
	chunkOut   []domain.Chunk
	cleanupErr error

	probeCalls   int
	copyCalls    int
	extractCalls int
	transcCalls  int
	chunkCalls   int
	cleanupCalls int
}

func (f *fakeAudio) Probe(string) (domain.AudioFile, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.probeCalls++
	return f.probeOut, nil
}
func (f *fakeAudio) CopyAudio(_ context.Context, in domain.AudioFile, _ string) (domain.AudioFile, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.copyCalls++
	out := f.copyOut
	if out.Path == "" {
		out = in
		out.IsTemp = true
		out.Complete = true
	}
	return out, nil
}
func (f *fakeAudio) ExtractAudio(_ context.Context, _ string, _ string) (domain.AudioFile, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.extractCalls++
	return f.extractOut, nil
}
func (f *fakeAudio) Transcode(_ context.Context, in domain.AudioFile, _ ports.TargetFormat, _ string) (domain.AudioFile, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.transcCalls++
	out := f.transcOut
	if out.Path == "" {
		out = in
		out.IsTemp = true
		out.Complete = true
	}
	return out, nil
}
func (f *fakeAudio) Chunk(_ context.Context, in domain.AudioFile, _ int64, _ string) ([]domain.Chunk, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.chunkCalls++
	if f.chunkOut != nil {
		return f.chunkOut, nil
	}
	return []domain.Chunk{{Path: in.Path, SizeBytes: in.SizeBytes, Complete: true}}, nil
}
func (f *fakeAudio) Cleanup(domain.AudioFile) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cleanupCalls++
	return f.cleanupErr
}

var _ ports.AudioProcessor = (*fakeAudio)(nil)

// fakeProviderFull is a complete Provider implementation for K6 tests.
type fakeProviderFull struct {
	id        domain.ProviderID
	models    []string
	caps      map[string]ports.ModelCapabilities
	maxUpload int64
	result    *domain.Result
	err       error
	transcribeFn func(ctx context.Context, audio domain.AudioFile, opts ports.ProviderOpts) (*domain.Result, error)
}

func (f *fakeProviderFull) ID() domain.ProviderID { return f.id }
func (f *fakeProviderFull) MaxUploadBytes() int64 { return f.maxUpload }
func (f *fakeProviderFull) Models() []string      { return f.models }
func (f *fakeProviderFull) DefaultModel() string {
	if len(f.models) > 0 {
		return f.models[0]
	}
	return ""
}
func (f *fakeProviderFull) Capabilities(m string) ports.ModelCapabilities {
	if c, ok := f.caps[m]; ok {
		return c
	}
	return ports.ModelCapabilities{}
}
func (f *fakeProviderFull) CheckKey(_ context.Context) error { return nil }
func (f *fakeProviderFull) Transcribe(ctx context.Context, a domain.AudioFile, o ports.ProviderOpts) (*domain.Result, error) {
	if f.transcribeFn != nil {
		return f.transcribeFn(ctx, a, o)
	}
	if f.err != nil {
		return nil, f.err
	}
	if f.result == nil {
		return nil, errors.New("fake provider not configured")
	}
	return f.result, nil
}

var _ ports.Provider = (*fakeProviderFull)(nil)

// fakeCache is a map-backed ResultCache.
type fakeCache struct {
	mu    sync.Mutex
	store map[string]*domain.Result
	saves int
}

func newFakeCache() *fakeCache {
	return &fakeCache{store: map[string]*domain.Result{}}
}

func (c *fakeCache) Lookup(path string, _ domain.ProviderID) (*domain.Result, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	r, ok := c.store[path]
	return r, ok, nil
}
func (c *fakeCache) Save(path string, r *domain.Result) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.store[path] = r
	c.saves++
	return nil
}

var _ ports.ResultCache = (*fakeCache)(nil)

// recordingWriter captures writes for assertion.
type recordingWriter struct {
	format domain.OutputFormat
	paths  []string
}

func (w *recordingWriter) Format() domain.OutputFormat { return w.format }
func (w *recordingWriter) Write(_ *domain.Result, dst string, _ domain.WriteOpts) error {
	w.paths = append(w.paths, dst)
	return nil
}

var _ ports.FormatWriter = (*recordingWriter)(nil)
