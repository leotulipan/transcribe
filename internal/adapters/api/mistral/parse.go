package mistral

import (
	"encoding/json"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// response mirrors Mistral Voxtral's transcription response.
// Voxtral returns text, language, duration, and segments.
// Result.Words stays empty (segment-level only in v1).
type response struct {
	Text     string    `json:"text"`
	Language string    `json:"language"`
	Duration float64   `json:"duration"`
	Segments []segment `json:"segments"`
}

type segment struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

func parse(data []byte, model string) (*domain.Result, error) {
	var resp response
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	res := &domain.Result{
		Provider: domain.ProviderMistral,
		Model:    model,
		Language: resp.Language,
		Text:     resp.Text,
		Duration: time.Duration(resp.Duration * float64(time.Second)),
		RawJSON:  data,
	}
	for _, s := range resp.Segments {
		res.Segments = append(res.Segments, domain.Segment{
			Text:  s.Text,
			Start: time.Duration(s.Start * float64(time.Second)),
			End:   time.Duration(s.End * float64(time.Second)),
		})
	}
	return res, nil
}
