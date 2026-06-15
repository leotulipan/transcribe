package audio

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

type ffprobeOutput struct {
	Format struct {
		Filename   string `json:"filename"`
		FormatName string `json:"format_name"`
		Duration   string `json:"duration"`
		Size       string `json:"size"`
	} `json:"format"`
	Streams []struct {
		CodecType string `json:"codec_type"`
		CodecName string `json:"codec_name"`
	} `json:"streams"`
}

func (f *FFmpeg) Probe(path string) (domain.AudioFile, error) {
	cmd := exec.CommandContext(context.Background(),
		f.ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path,
	)
	hideConsole(cmd)
	out, err := cmd.Output()
	if err != nil {
		return domain.AudioFile{}, fmt.Errorf("ffprobe: %w", err)
	}
	var p ffprobeOutput
	if err := json.Unmarshal(out, &p); err != nil {
		return domain.AudioFile{}, err
	}
	af := domain.AudioFile{Path: path}
	af.Container = pickContainer(p.Format.FormatName, path)
	for _, s := range p.Streams {
		if s.CodecType == "audio" {
			af.Codec = s.CodecName
			break
		}
	}
	if sec, err := strconv.ParseFloat(p.Format.Duration, 64); err == nil {
		af.Duration = time.Duration(sec * float64(time.Second))
	}
	if n, err := strconv.ParseInt(p.Format.Size, 10, 64); err == nil {
		af.SizeBytes = n
	} else if info, err := os.Stat(path); err == nil {
		af.SizeBytes = info.Size()
	}
	return af, nil
}

// pickContainer picks the most useful container name. ffprobe's format_name is
// a comma-separated list (e.g. "mov,mp4,m4a,3gp"). Prefer the file extension
// when it appears in the list; otherwise take the first entry.
func pickContainer(formatName, path string) string {
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(path)), ".")
	parts := strings.Split(formatName, ",")
	if ext != "" {
		for _, p := range parts {
			if p == ext {
				return ext
			}
		}
	}
	if len(parts) > 0 {
		return parts[0]
	}
	return ""
}
