package services

import (
	"context"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

// codecAccepted reports whether the file can be sent as-is, with no container
// conversion. An AcceptedInputs token may name either a real audio codec
// (e.g. "flac", "aac", "opus") or a container/format (e.g. "wav", "mp4",
// "m4a", "ogg", "webm"). ffprobe reports the real codec in af.Codec and the
// container in af.Container, so a format token like "wav" never equals
// af.Codec ("pcm_s16le") — it must be matched against af.Container. Failing to
// do so re-encodes WAV/M4A/MP4 inputs needlessly even though every provider
// accepts them directly.
func codecAccepted(caps ports.ModelCapabilities, af domain.AudioFile) bool {
	for _, in := range caps.AcceptedInputs {
		// Explicit container constraint: codec (if set) and container must
		// both match.
		if in.Container != "" {
			if (in.Codec == "" || in.Codec == af.Codec) && in.Container == af.Container {
				return true
			}
			continue
		}
		// Blank container — accept as-is when either:
		//  - the token names this file's container/format directly (the
		//    provider takes the whole container, e.g. wav, mp4, m4a, ogg), or
		//  - the file is a simple audio-only container whose codec name equals
		//    its container (e.g. mp3/mp3, flac/flac).
		if in.Codec == af.Container {
			return true
		}
		if in.Codec == af.Codec && af.Codec == af.Container {
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
	opts ports.PrepareOpts,
) (domain.AudioFile, error) {
	// UseInput: power-user bypass — send source as-is no matter what.
	if opts.UseInput {
		return src, nil
	}

	// Compute the effective size ceiling for the as-is path. When the user has
	// set a size threshold (> 0), that value widens the path: files up to the
	// threshold are sent as-is even if they exceed the provider's own maxBytes.
	// The provider may reject the upload; that is the user's stated preference.
	effectiveMax := maxBytes
	if opts.SizeThresholdBytes > effectiveMax {
		effectiveMax = opts.SizeThresholdBytes
	}

	// 5a: as-is
	if codecAccepted(caps, src) && src.SizeBytes <= effectiveMax {
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
