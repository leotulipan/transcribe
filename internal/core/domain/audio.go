package domain

import "time"

// AudioFile describes an audio (or audio-bearing) file on disk.
type AudioFile struct {
	Path      string
	SizeBytes int64
	Duration  time.Duration
	Container string  // mp4, m4a, wav, mp3, flac, ogg, webm
	Codec     string  // aac, mp3, pcm_s16le, flac, opus
	IsTemp    bool    // managed temp file; cleanup eligible
	Complete  bool    // ffmpeg returned 0 and file fully on disk
	Chunks    []Chunk
}

// AudioFormat describes an accepted container/codec combination.
// An empty Container means "any container is fine as long as Codec matches".
type AudioFormat struct {
	Container string
	Codec     string
}

// Chunk is one slice of a larger audio file produced for size-limited APIs.
type Chunk struct {
	Path        string
	StartOffset time.Duration
	SizeBytes   int64
	Complete    bool
}
