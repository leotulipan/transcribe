package domain

import "time"

// Request describes a single transcription job submitted to the service.
type Request struct {
	InputPath   string
	Provider    ProviderID
	Model       string         // "" = provider default
	Language    string         // ISO-639-1; "" = auto-detect
	Formats     []OutputFormat
	OutputDir   string         // "" = next to input
	DaVinciOpts *DaVinciOptions
	UseCache    bool
}

// Result is the normalized output every provider produces.
type Result struct {
	Provider   ProviderID
	Model      string
	Language   string
	Text       string
	Confidence float64
	Words      []Word
	Segments   []Segment
	Speakers   []Speaker      // empty in v1
	Duration   time.Duration
	SourcePath string
	RawJSON    []byte         // pristine provider response (JSON array if merged)
}

type Word struct {
	Text       string
	Start, End time.Duration
	Confidence float64
}

type Segment struct {
	Text       string
	Start, End time.Duration
	SpeakerID  string
}

type Speaker struct {
	ID, Label string
}

type DaVinciOptions struct {
	SilentPortionThreshold time.Duration
	PaddingStart           time.Duration
	FillerWords            []string
}

// DefaultFillerWords is what DaVinciOptions.FillerWords defaults to when empty.
var DefaultFillerWords = []string{"um", "uh", "ähm", "äh", "hm", "hmm"}
