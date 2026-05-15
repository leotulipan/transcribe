package audio

import (
	"encoding/json"
	"os"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

const metaSchema = 1

// MetaInfo describes a cached intermediate audio file so the next run can
// decide whether it is reusable.
type MetaInfo struct {
	SchemaVersion   int               `json:"schema_version"`
	Operation       string            `json:"operation"` // "copy" | "transcode"
	SourcePath      string            `json:"source_path"`
	SourceSize      int64             `json:"source_size"`
	SourceMTimeUnix int64             `json:"source_mtime_unix"`
	TargetCodec     string            `json:"target_codec"`
	TargetContainer string            `json:"target_container"`
	MaxBytesBudget  int64             `json:"max_bytes_budget"`
	Provider        domain.ProviderID `json:"provider"`
	Model           string            `json:"model"`
}

func metaPath(intermediate string) string { return intermediate + ".meta.json" }

func WriteMeta(intermediate string, m MetaInfo) error {
	m.SchemaVersion = metaSchema
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(metaPath(intermediate), data, 0o644)
}

func ReadMeta(intermediate string) (MetaInfo, error) {
	var m MetaInfo
	data, err := os.ReadFile(metaPath(intermediate))
	if err != nil {
		return m, err
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return m, err
	}
	return m, nil
}
