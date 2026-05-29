package elevenlabs

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/leotulipan/transcribe/internal/adapters/api/internal/discover"
	"github.com/leotulipan/transcribe/internal/core/domain"
)

// sttPrefix is the family prefix for ElevenLabs speech-to-text models. The
// public /v1/models endpoint returns every TTS / STS voice model alongside
// any STT models, with no can_do_speech_to_text flag to filter on. Until
// upstream surfaces one, the prefix is the only reliable signal.
const sttPrefix = "scribe_"

// DiscoverModels lists models via GET /v1/models and filters them to the
// STT family. The endpoint returns a flat JSON array of model objects, each
// carrying a `model_id`. ElevenLabs exposes capability flags for TTS / STS
// but no can_do_speech_to_text flag, so we keep only IDs that start with
// "scribe_" — the documented STT prefix. If the upstream response contains
// zero scribe_* entries (e.g. transient outage or schema change), fall back
// to the hardcoded Models() list so the dropdown is never empty.
func (c *Client) DiscoverModels(ctx context.Context) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, checkKeyTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", c.endpoint+modelsPath, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("xi-api-key", c.apiKey)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, &domain.ErrProvider{Provider: domain.ProviderElevenLabs, Cause: err}
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode/100 != 2 {
		return nil, &domain.ErrProvider{
			Provider:   domain.ProviderElevenLabs,
			StatusCode: resp.StatusCode,
			Cause:      fmt.Errorf("%s", string(body)),
		}
	}
	var payload []struct {
		ModelID string `json:"model_id"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("elevenlabs discover: parse: %w", err)
	}
	ids := make([]string, 0, len(payload))
	for _, m := range payload {
		if strings.HasPrefix(m.ModelID, sttPrefix) {
			ids = append(ids, m.ModelID)
		}
	}
	if len(ids) == 0 {
		ids = Models()
	}
	return discover.SortUnique(ids), nil
}
