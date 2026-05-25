package domain

import "time"

// Request describes a single transcription job submitted to the service.
type Request struct {
	InputPath        string
	Provider         ProviderID
	Model            string         // "" = provider default
	Language         string         // ISO-639-1; "" = auto-detect
	Formats          []OutputFormat
	OutputDir        string         // "" = next to input
	DaVinciOpts      *DaVinciOptions
	UseCache         bool
	MaxCharsPerLine  int            // 0 = no wrapping; positive = max chars per rendered subtitle line
	WordsPerSubtitle int            // 0 = format default (7); positive overrides groupWords maxWords
	StartHour        int            // hour offset added to every timecode in SRT / DaVinci output
	SpeakerLabels    bool           // request diarization from the provider
	NumSpeakers      int            // 0 = unset; 1..32 valid; only meaningful when SpeakerLabels=true
	KeyTerms         []string       // comma-parsed keyterms; empty = unset
	SpeechModels     []string       // fallback speech model list; empty = unset

	// Audio pipeline knobs (Phase 5d).
	SizeThresholdBytes    int64 // 0 = use provider maxBytes only; >0 widens the as-is path past provider max
	ChunkLengthSec        int   // 0 = derive from byte budget; >0 overrides chunk duration
	OverlapSec            int   // 0 = no overlap; >0 = each chunk starts this many seconds before the nominal boundary
	UseInput              bool  // bypass conversion entirely — send source file as-is
	UsePCM                bool  // transcode to PCM WAV (pcm_s16le) instead of the preferred codec
	KeepIntermediates     bool  // retain all temp files even on success
	KeepFLACIntermediates bool  // retain temp files whose codec/container is flac

	// I/O & workflow knobs (Phase 5e).
	UseJSONInput    bool // treat InputPath as a pre-saved sidecar JSON; skip API call entirely
	SaveCleanedJSON bool // persist the normalized domain.Result JSON even when UseCache=false
}

// WriteOpts carries per-write rendering knobs that are derived from Request
// and forwarded to FormatWriter.Write. Separating these from domain.Result
// keeps provider output clean of delivery concerns.
type WriteOpts struct {
	MaxCharsPerLine  int  // 0 = no wrapping
	SpeakerLabels    bool // prefix each subtitle block with [Speaker X]:
	WordsPerSubtitle int  // 0 = use format default (7); positive overrides groupWords maxWords
	StartHour        int  // hour offset added to every timecode in SRT / DaVinci output
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
	PaddingEnd             time.Duration // shrink each word's End symmetrically to PaddingStart
	FillerWords            []string
	RemoveFillers          bool
	SuppressFillerLines    bool
	SuppressPauses         bool // when true, no (...) markers are inserted even on long gaps
	FPS                    float64
	FPSOffsetStart         int // frames added to snapped Start; CLI default is -1 (appear one frame early)
	FPSOffsetEnd           int // frames added to snapped End; CLI default is 0
}

// DefaultFillerWords is what DaVinciOptions.FillerWords defaults to when empty.
var DefaultFillerWords = []string{"um", "uh", "ähm", "äh", "hm", "hmm"}
