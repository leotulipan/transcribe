package gemini

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// generateContentResponse mirrors Gemini's generateContent response envelope.
// Gemini returns free-form text; no word or segment timestamps are available.
type generateContentResponse struct {
	Candidates []candidate `json:"candidates"`
}

type candidate struct {
	Content      content `json:"content"`
	FinishReason string  `json:"finishReason"`
}

type content struct {
	Parts []part `json:"parts"`
}

type part struct {
	Text string `json:"text"`
}

func parse(data []byte, model string) (*domain.Result, error) {
	var resp generateContentResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("gemini: no candidates in response")
	}
	var sb strings.Builder
	for _, p := range resp.Candidates[0].Content.Parts {
		sb.WriteString(p.Text)
	}
	text := strings.TrimSpace(sb.String())
	return &domain.Result{
		Provider: domain.ProviderGemini,
		Model:    model,
		Text:     text,
		RawJSON:  data,
		// No Words or Segments — Gemini is text-only for transcription.
	}, nil
}
