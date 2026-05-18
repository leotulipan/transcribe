package services

import (
	"context"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// codecAccepted reports whether (container, codec) is directly accepted by the
// model — i.e., the file can be sent as-is without any container conversion.
// A blank Container in the accepted entry means "accept if the source is a
// simple audio-only file where the container name equals the codec name (e.g.
// mp3/mp3, flac/flac)". An explicit Container requires an exact match.
func codecAccepted(caps ports.ModelCapabilities, af domain.AudioFile) bool {
	for _, in := range caps.AcceptedInputs {
		if in.Codec != "" && in.Codec != af.Codec {
			continue
		}
		// Explicit container match
		if in.Container != "" && in.Container == af.Container {
			return true
		}
		// Blank container: accept when the source is a simple audio-only container
		// (codec == container, e.g., mp3/mp3, flac/flac, ogg/ogg).
		if in.Container == "" && af.Codec == af.Container {
			return true
		}
	}
	return false
}

// codecOnlyAccepted reports whether the codec alone is accepted (any container).
func codecOnlyAccepted(caps ports.ModelCapabilities, codec string) bool {
	for _, in := range caps.AcceptedInputs {
		if in.Codec == codec {
			return true
		}
	}
	return false
}

// prepare implements the copy-first decision tree (spec §6.3 step 5).
func prepare(
	ctx context.Context,
	audio ports.AudioProcessor,
	src domain.AudioFile,
	caps ports.ModelCapabilities,
	maxBytes int64,
	workDir string,
	transcodeTarget ports.TargetFormat,
) (domain.AudioFile, error) {
	// 5a: as-is
	if codecAccepted(caps, src) && src.SizeBytes <= maxBytes {
		return src, nil
	}
	// 5b: stream copy when the codec is accepted but container isn't (or video wrapper)
	if codecOnlyAccepted(caps, src.Codec) {
		out, err := audio.CopyAudio(ctx, src, workDir)
		if err == nil && out.SizeBytes <= maxBytes {
			return out, nil
		}
		// fall through to transcode on either error or "still too big"
	}
	// 5c: transcode
	return audio.Transcode(ctx, src, transcodeTarget, workDir)
}
