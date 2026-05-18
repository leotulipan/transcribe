package mistral

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/leotulipan/transcribe/internal/adapters/api/internal/discover"
	"github.com/leotulipan/transcribe/internal/core/domain"
)

// DiscoverModels lists models via GET /v1/models. Same OpenAI-compatible
// envelope: {"data": [{"id": "..."}]}.
func (c *Client) DiscoverModels(ctx context.Context) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, checkKeyTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", c.endpoint+modelsPath, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, &domain.ErrProvider{Provider: domain.ProviderMistral, Cause: err}
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode/100 != 2 {
		return nil, &domain.ErrProvider{
			Provider:   domain.ProviderMistral,
			StatusCode: resp.StatusCode,
			Cause:      fmt.Errorf("%s", string(body)),
		}
	}
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("mistral discover: parse: %w", err)
	}
	ids := make([]string, 0, len(payload.Data))
	for _, m := range payload.Data {
		ids = append(ids, m.ID)
	}
	return discover.SortUnique(ids), nil
}
