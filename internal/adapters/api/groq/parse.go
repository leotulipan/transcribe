package groq

import (
	"encoding/json"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

type response struct {
	Text     string    `json:"text"`
	Language string    `json:"language"`
	Duration float64   `json:"duration"`
	Segments []segment `json:"segments"`
	Words    []word    `json:"words"`
}

type segment struct {
	ID         int     `json:"id"`
	Start      float64 `json:"start"`
	End        float64 `json:"end"`
	Text       string  `json:"text"`
	AvgLogprob float64 `json:"avg_logprob"`
}

type word struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

func parse(data []byte, model string) (*domain.Result, error) {
	var resp response
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	res := &domain.Result{
		Provider: domain.ProviderGroq,
		Model:    model,
		Language: resp.Language,
		Text:     resp.Text,
		Duration: time.Duration(resp.Duration * float64(time.Second)),
		RawJSON:  data,
	}
	for _, w := range resp.Words {
		res.Words = append(res.Words, domain.Word{
			Text:  w.Word,
			Start: time.Duration(w.Start * float64(time.Second)),
			End:   time.Duration(w.End * float64(time.Second)),
		})
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
