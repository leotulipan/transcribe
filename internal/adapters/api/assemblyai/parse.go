package assemblyai

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// transcriptResponse mirrors AssemblyAI's GET /v2/transcript/{id} response.
// word start/end are milliseconds (integers).
type transcriptResponse struct {
	ID            string  `json:"id"`
	Status        string  `json:"status"`
	Error         string  `json:"error"`
	Text          string  `json:"text"`
	LanguageCode  string  `json:"language_code"`
	AudioDuration float64 `json:"audio_duration"` // seconds
	Words         []aWord `json:"words"`
}

type aWord struct {
	Text  string `json:"text"`
	Start int64  `json:"start"` // milliseconds
	End   int64  `json:"end"`   // milliseconds
}

func parse(data []byte, model string) (*domain.Result, error) {
	var resp transcriptResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	if resp.Status == "error" {
		return nil, fmt.Errorf("assemblyai transcription error: %s", resp.Error)
	}
	res := &domain.Result{
		Provider: domain.ProviderAssemblyAI,
		Model:    model,
		Language: resp.LanguageCode,
		Text:     resp.Text,
		Duration: time.Duration(resp.AudioDuration * float64(time.Second)),
		RawJSON:  data,
	}
	for _, w := range resp.Words {
		res.Words = append(res.Words, domain.Word{
			Text:  w.Text,
			Start: time.Duration(w.Start) * time.Millisecond,
			End:   time.Duration(w.End) * time.Millisecond,
		})
	}
	return res, nil
}
