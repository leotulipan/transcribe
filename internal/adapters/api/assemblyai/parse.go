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
	SpeechModel   string  `json:"speech_model"`   // model actually used; echoed by the API
	AudioDuration float64 `json:"audio_duration"` // seconds
	Words         []aWord `json:"words"`
}

type aWord struct {
	Text    string `json:"text"`
	Start   int64  `json:"start"`   // milliseconds
	End     int64  `json:"end"`     // milliseconds
	Speaker string `json:"speaker"` // populated when speaker_labels=true
}

func parse(data []byte, model string) (*domain.Result, error) {
	var resp transcriptResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	if resp.Status == "error" {
		return nil, fmt.Errorf("assemblyai transcription error: %s", resp.Error)
	}
	// In auto mode we never named a model, so report the one AssemblyAI ran.
	// The echo can carry legacy labels (older responses say "best"); we record
	// it verbatim as information and never feed it back into a request.
	if model == AutoModel && resp.SpeechModel != "" {
		model = resp.SpeechModel
	}
	res := &domain.Result{
		Provider: domain.ProviderAssemblyAI,
		Model:    model,
		Language: resp.LanguageCode,
		Text:     resp.Text,
		Duration: time.Duration(resp.AudioDuration * float64(time.Second)),
		RawJSON:  data,
	}
	seen := map[string]bool{}
	for _, w := range resp.Words {
		res.Words = append(res.Words, domain.Word{
			Text:    w.Text,
			Start:   time.Duration(w.Start) * time.Millisecond,
			End:     time.Duration(w.End) * time.Millisecond,
			Speaker: w.Speaker,
		})
		if w.Speaker != "" && !seen[w.Speaker] {
			seen[w.Speaker] = true
			res.Speakers = append(res.Speakers, domain.Speaker{ID: w.Speaker})
		}
	}
	return res, nil
}
