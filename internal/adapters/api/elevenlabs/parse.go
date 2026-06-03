package elevenlabs

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/leotulipan/transcribe/internal/core/domain"
)

// response mirrors ElevenLabs speech-to-text response.
// language_code is ISO-639-3 (e.g. "eng"); passed through verbatim.
// Words include both "word" and "spacing" types; spacing entries are filtered.
type response struct {
	LanguageCode        string  `json:"language_code"`
	LanguageProbability float64 `json:"language_probability"`
	Text                string  `json:"text"`
	Words               []word  `json:"words"`
}

type word struct {
	Text      string  `json:"text"`
	Start     float64 `json:"start"`
	End       float64 `json:"end"`
	Type      string  `json:"type"`       // "word" or "spacing"
	SpeakerID string  `json:"speaker_id"` // populated when diarize=true
}

func parse(data []byte, model string) (*domain.Result, error) {
	var resp response
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	res := &domain.Result{
		Provider: domain.ProviderElevenLabs,
		Model:    model,
		Language: resp.LanguageCode,
		Text:     resp.Text,
		RawJSON:  data,
	}
	seen := map[string]bool{}
	for _, w := range resp.Words {
		if w.Type != "word" {
			continue // skip spacing entries
		}
		spk := normalizeSpeakerID(w.SpeakerID)
		res.Words = append(res.Words, domain.Word{
			Text:    w.Text,
			Start:   time.Duration(w.Start * float64(time.Second)),
			End:     time.Duration(w.End * float64(time.Second)),
			Speaker: spk,
		})
		if spk != "" && !seen[spk] {
			seen[spk] = true
			res.Speakers = append(res.Speakers, domain.Speaker{ID: spk})
		}
	}
	return res, nil
}

// normalizeSpeakerID converts provider speaker ids to a clean token. ElevenLabs
// returns ids like "speaker_0"; we strip the "speaker_"/"speaker " prefix so the
// formatters render "[Speaker 0]" rather than "[Speaker speaker_0]". Already-clean
// ids (e.g. "A", "0") pass through unchanged.
func normalizeSpeakerID(id string) string {
	id = strings.TrimSpace(id)
	low := strings.ToLower(id)
	for _, p := range []string{"speaker_", "speaker "} {
		if strings.HasPrefix(low, p) {
			return strings.TrimSpace(id[len(p):])
		}
	}
	return id
}
