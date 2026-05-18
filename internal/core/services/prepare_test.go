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

	out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"})
	require.NoError(t, err)
	require.Equal(t, "in.mp3", out.Path)
	require.False(t, out.IsTemp)
	require.Equal(t, 0, audio.copyCalls)
	require.Equal(t, 0, audio.transcCalls)
}

func TestPrepare_CopyWhenAcceptedButContainerCantBeStreamed(t *testing.T) {
	// mp4 container containing AAC — AAC is accepted but the mp4 box around
	// it must be stream-copied into m4a.
	audio := &fakeAudio{copyOut: domain.AudioFile{Path: "out.m4a", Codec: "aac", Container: "m4a", IsTemp: true, Complete: true, SizeBytes: 200}}
	src := domain.AudioFile{Path: "in.mp4", Codec: "aac", Container: "mp4", SizeBytes: 500}
	caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{{Codec: "aac"}}}

	out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"})
	require.NoError(t, err)
	require.Equal(t, "out.m4a", out.Path)
	require.Equal(t, 1, audio.copyCalls)
	require.Equal(t, 0, audio.transcCalls)
}

func TestPrepare_TranscodeWhenSourceTooLarge(t *testing.T) {
	audio := &fakeAudio{transcOut: domain.AudioFile{Path: "out.mp3", Codec: "mp3", Container: "mp3", IsTemp: true, Complete: true, SizeBytes: 500}}
	src := domain.AudioFile{Path: "in.wav", Codec: "pcm_s16le", Container: "wav", SizeBytes: 10000}
	caps := ports.ModelCapabilities{AcceptedInputs: []domain.AudioFormat{{Codec: "mp3"}}}

	out, err := prepare(context.Background(), audio, src, caps, 1024, "/tmp", ports.TargetFormat{Codec: "mp3"})
	require.NoError(t, err)
	require.Equal(t, "out.mp3", out.Path)
	require.Equal(t, 1, audio.transcCalls)
}
