package services

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

func TestPrepare_AsIsWhenAcceptedAndSmallEnough(t *testing.T) {
	audio := &fakeAudio{}
	src := domain.AudioFile{Path: "in.mp3", Codec: "mp3", Container: "mp3", SizeBytes: 100}
	caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}}

	out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"}, ports.PrepareOpts{})
	require.NoError(t, err)
	require.Equal(t, "in.mp3", out.Path)
	require.False(t, out.IsTemp)
	require.Equal(t, 0, audio.copyCalls)
	require.Equal(t, 0, audio.transcCalls)
}

func TestConversionDecisionMsg(t *testing.T) {
	src := domain.AudioFile{Path: "in.wav", Codec: "pcm_s16le", Container: "wav"}

	// As-is: prepared is the same file.
	require.Contains(t, conversionDecisionMsg(src, src), "as-is")

	// Stream-copy: same codec, different container/path.
	copied := domain.AudioFile{Path: "out.m4a", Codec: "pcm_s16le", Container: "wav-copy"}
	require.Contains(t, conversionDecisionMsg(src, copied), "stream-copied")

	// Transcode: different codec.
	transcoded := domain.AudioFile{Path: "out.flac", Codec: "flac", Container: "flac"}
	msg := conversionDecisionMsg(src, transcoded)
	require.Contains(t, msg, "transcoded")
	require.Contains(t, msg, "flac")
}

func TestPrepare_WavAcceptedAsIs(t *testing.T) {
	// ElevenLabs/OpenAI accept WAV directly. A probed WAV reports codec
	// "pcm_s16le" with container "wav", while the accepted-input token is the
	// format name "wav". It must be sent as-is — never transcoded.
	audio := &fakeAudio{}
	src := domain.AudioFile{Path: "in.wav", Codec: "pcm_s16le", Container: "wav", SizeBytes: 100}
	caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{
		{Codec: "mp3"}, {Codec: "wav"}, {Codec: "flac"},
	}}

	out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "flac"}, ports.PrepareOpts{})
	require.NoError(t, err)
	require.Equal(t, "in.wav", out.Path)
	require.False(t, out.IsTemp)
	require.Equal(t, 0, audio.copyCalls)
	require.Equal(t, 0, audio.transcCalls)
}

func TestPrepare_M4aAcceptedAsIs(t *testing.T) {
	// .m4a wraps AAC in an MP4/M4A box; ffprobe reports codec "aac",
	// container "m4a". A provider listing the "m4a" format accepts it
	// directly — no transcode, no rewrap.
	audio := &fakeAudio{}
	src := domain.AudioFile{Path: "in.m4a", Codec: "aac", Container: "m4a", SizeBytes: 100}
	caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{
		{Codec: "mp3"}, {Codec: "mp4"}, {Codec: "m4a"}, {Codec: "wav"},
	}}

	out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "flac"}, ports.PrepareOpts{})
	require.NoError(t, err)
	require.Equal(t, "in.m4a", out.Path)
	require.Equal(t, 0, audio.copyCalls)
	require.Equal(t, 0, audio.transcCalls)
}

func TestPrepare_CopyWhenAcceptedButContainerCantBeStreamed(t *testing.T) {
	// mp4 container containing AAC — AAC is accepted but the mp4 box around
	// it must be stream-copied into m4a.
	audio := &fakeAudio{copyOut: domain.AudioFile{Path: "out.m4a", Codec: "aac", Container: "m4a", IsTemp: true, Complete: true, SizeBytes: 200}}
	src := domain.AudioFile{Path: "in.mp4", Codec: "aac", Container: "mp4", SizeBytes: 500}
	caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{{Codec: "aac"}}}

	out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"}, ports.PrepareOpts{})
	require.NoError(t, err)
	require.Equal(t, "out.m4a", out.Path)
	require.Equal(t, 1, audio.copyCalls)
	require.Equal(t, 0, audio.transcCalls)
}

func TestPrepare_TranscodeWhenSourceTooLarge(t *testing.T) {
	audio := &fakeAudio{transcOut: domain.AudioFile{Path: "out.mp3", Codec: "mp3", Container: "mp3", IsTemp: true, Complete: true, SizeBytes: 500}}
	src := domain.AudioFile{Path: "in.wav", Codec: "pcm_s16le", Container: "wav", SizeBytes: 10000}
	caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}}

	out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"}, ports.PrepareOpts{})
	require.NoError(t, err)
	require.Equal(t, "out.mp3", out.Path)
	require.Equal(t, 1, audio.transcCalls)
}

func TestPrepare_UseInputBypassesAllChecks(t *testing.T) {
	// UseInput=true: even an incompatible big file is returned as-is.
	audio := &fakeAudio{}
	src := domain.AudioFile{Path: "in.ogg", Codec: "opus", Container: "ogg", SizeBytes: 999_999_999}
	caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}}

	out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"}, ports.PrepareOpts{UseInput: true})
	require.NoError(t, err)
	require.Equal(t, "in.ogg", out.Path)
	require.Equal(t, 0, audio.copyCalls)
	require.Equal(t, 0, audio.transcCalls)
}

func TestPrepare_SizeThresholdWidensAsIsPath(t *testing.T) {
	// File is codec-compatible, exceeds provider maxBytes (1024), but is below
	// the user threshold (5000). It should be returned as-is.
	audio := &fakeAudio{}
	src := domain.AudioFile{Path: "in.mp3", Codec: "mp3", Container: "mp3", SizeBytes: 3000}
	caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}}

	out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"}, ports.PrepareOpts{SizeThresholdBytes: 5000})
	require.NoError(t, err)
	require.Equal(t, "in.mp3", out.Path)
	require.Equal(t, 0, audio.copyCalls)
	require.Equal(t, 0, audio.transcCalls)
}

func TestPrepare_SizeThresholdDoesNotApplyWhenIncompatible(t *testing.T) {
	// Codec is not accepted: even with a generous threshold, we must transcode.
	audio := &fakeAudio{transcOut: domain.AudioFile{Path: "out.mp3", Codec: "mp3", Container: "mp3", IsTemp: true, Complete: true, SizeBytes: 800}}
	src := domain.AudioFile{Path: "in.ogg", Codec: "opus", Container: "ogg", SizeBytes: 500}
	caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}}

	out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"}, ports.PrepareOpts{SizeThresholdBytes: 999_999})
	require.NoError(t, err)
	require.Equal(t, "out.mp3", out.Path)
	require.Equal(t, 1, audio.transcCalls)
}
