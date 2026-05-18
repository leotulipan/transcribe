package gemini

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

// DiscoverModels lists models via GET /v1beta/models. Gemini returns names
// like "models/gemini-2.5-flash"; we strip the "models/" prefix so the
// returned IDs match what callers pass back into ProviderOpts.Model.
func (c *Client) DiscoverModels(ctx context.Context) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, checkKeyTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", c.endpoint+modelsPath, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-goog-api-key", c.apiKey)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, &domain.ErrProvider{Provider: domain.ProviderGemini, Cause: err}
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode/100 != 2 {
		return nil, &domain.ErrProvider{
			Provider:   domain.ProviderGemini,
			StatusCode: resp.StatusCode,
			Cause:      fmt.Errorf("%s", string(body)),
		}
	}
	var payload struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("gemini discover: parse: %w", err)
	}
	ids := make([]string, 0, len(payload.Models))
	for _, m := range payload.Models {
		ids = append(ids, strings.TrimPrefix(m.Name, "models/"))
	}
	return discover.SortUnique(ids), nil
}
