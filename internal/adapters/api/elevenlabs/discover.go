package elevenlabs

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/leotulipan/transcribe/internal/adapters/api/internal/discover"
	"github.com/leotulipan/transcribe/internal/core/domain"
)

// DiscoverModels lists models via GET /v1/models. The response is a flat
// JSON array of model objects, each carrying a `model_id`. The schema has
// capability flags (can_do_text_to_speech, can_do_voice_conversion) but no
// explicit STT flag yet, so we return every model_id and let the user pick.
// If a future API revision adds a can_do_speech_to_text flag (or surfaces
// scribe_v1 here), filter at that point.
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
		ids = append(ids, m.ModelID)
	}
	return discover.SortUnique(ids), nil
}
