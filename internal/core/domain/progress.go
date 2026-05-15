package domain

import "time"

type Stage int

const (
	StageProbing Stage = iota
	StageExtracting
	StageCompressing
	StageChunking
	StageUploading
	StageTranscribing
	StageParsing
	StageWriting
	StageDone
)

func (s Stage) String() string {
	switch s {
	case StageProbing:      return "probing"
	case StageExtracting:   return "extracting"
	case StageCompressing:  return "compressing"
	case StageChunking:     return "chunking"
	case StageUploading:    return "uploading"
	case StageTranscribing: return "transcribing"
	case StageParsing:      return "parsing"
	case StageWriting:      return "writing"
	case StageDone:         return "done"
	}
	return "unknown"
}

// ProgressEvent flows from the service to UIs over Job.Progress().
type ProgressEvent struct {
	Stage   Stage
	Message string
	Percent float64        // -1 when not estimable
	Elapsed time.Duration
}
