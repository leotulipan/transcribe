package services

import (
	"os"
	"path/filepath"

	"github.com/leotulipan/transcribe/internal/adapters/audio"
	"github.com/leotulipan/transcribe/internal/core/domain"
)

// lookupIntermediate scans workDir for a meta.json sidecar matching the
// (source, provider, model, budget, target-codec) tuple and returns the
// AudioFile pointing at it. Returns nil on no match.
func lookupIntermediate(workDir string, src domain.AudioFile, srcMTime int64, p domain.ProviderID, model string, budget int64, targetCodec string) *domain.AudioFile {
	entries, err := os.ReadDir(workDir)
	if err != nil {
		return nil
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if filepath.Ext(name) == ".json" {
			continue
		}
		full := filepath.Join(workDir, name)
		m, err := audio.ReadMeta(full)
		if err != nil {
			continue
		}
		if m.Provider != p || m.Model != model || m.TargetCodec != targetCodec {
			continue
		}
		if m.SourceSize != src.SizeBytes || m.MaxBytesBudget != budget {
			continue
		}
		if m.SourceMTimeUnix != 0 && srcMTime != 0 && m.SourceMTimeUnix != srcMTime {
			continue
		}
		info, err := os.Stat(full)
		if err != nil {
			continue
		}
		return &domain.AudioFile{
			Path:      full,
			SizeBytes: info.Size(),
			Container: m.TargetContainer,
			Codec:     m.TargetCodec,
			IsTemp:    true,
			Complete:  true,
		}
	}
	return nil
}
