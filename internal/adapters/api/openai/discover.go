package openai

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

// isSTTModel reports whether an OpenAI model id is speech-to-text capable.
// OpenAI's /v1/models returns chat, embedding, image and TTS models too; only
// the whisper / *-transcribe families do transcription.
func isSTTModel(id string) bool {
	return strings.Contains(id, "whisper") || strings.Contains(id, "transcribe")
}

// DiscoverModels lists models via GET /v1/models. OpenAI returns every model
// the key has access to (chat, embeddings, TTS, images…), so we filter to the
// STT models. If none match, fall back to the hardcoded list so the picker is
// never empty.
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
		return nil, &domain.ErrProvider{Provider: domain.ProviderOpenAI, Cause: err}
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode/100 != 2 {
		return nil, &domain.ErrProvider{
			Provider:   domain.ProviderOpenAI,
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
		return nil, fmt.Errorf("openai discover: parse: %w", err)
	}
	ids := make([]string, 0, len(payload.Data))
	for _, m := range payload.Data {
		if isSTTModel(m.ID) {
			ids = append(ids, m.ID)
		}
	}
	if len(ids) == 0 {
		return Models(), nil
	}
	return discover.SortUnique(ids), nil
}
