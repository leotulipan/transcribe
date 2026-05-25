package domain

import "time"

// Request describes a single transcription job submitted to the service.
type Request struct {
	InputPath       string
	Provider        ProviderID
	Model           string         // "" = provider default
	Language        string         // ISO-639-1; "" = auto-detect
	Formats         []OutputFormat
	OutputDir       string         // "" = next to input
	DaVinciOpts     *DaVinciOptions
	UseCache        bool
	MaxCharsPerLine int            // 0 = no wrapping; positive = max chars per rendered subtitle line
	SpeakerLabels   bool           // request diarization from the provider
}

// WriteOpts carries per-write rendering knobs that are derived from Request
// and forwarded to FormatWriter.Write. Separating these from domain.Result
// keeps provider output clean of delivery concerns.
type WriteOpts struct {
	MaxCharsPerLine int  // 0 = no wrapping
	SpeakerLabels   bool // prefix each subtitle block with [Speaker X]:
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
	Speaker    string // raw speaker ID from provider (e.g. "A", "B"); empty when diarization not requested
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
	RemoveFillers          bool
	SuppressFillerLines    bool
}

// DefaultFillerWords is what DaVinciOptions.FillerWords defaults to when empty.
var DefaultFillerWords = []string{"um", "uh", "ähm", "äh", "hm", "hmm"}
