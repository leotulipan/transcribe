package assemblyai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/leotulipan/transcribe/internal/adapters/api/internal/retry"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

const (
	defaultEndpoint = "https://api.assemblyai.com"
	maxUploadBytes  = 200 * 1024 * 1024
	uploadPath      = "/v2/upload"
	transcriptPath  = "/v2/transcript"
	checkKeyPath    = "/v2/transcript?limit=1"
	requestTimeout  = 10 * time.Minute
	checkKeyTimeout = 10 * time.Second
)

// pollInterval is the delay between transcript status polls.
// Set to a small value in tests to avoid slow tests.
var pollInterval = 1 * time.Second

// Client implements ports.Provider against AssemblyAI's two-step upload+poll flow.
// Auth: Authorization: <api_key> (no Bearer prefix).
type Client struct {
	apiKey   string
	endpoint string
	http     *http.Client
}

func New(apiKey string, h *http.Client) *Client {
	return NewWithEndpoint(apiKey, defaultEndpoint, h)
}

func NewWithEndpoint(apiKey, endpoint string, h *http.Client) *Client {
	if h == nil {
		h = &http.Client{Timeout: requestTimeout}
	}
	return &Client{apiKey: apiKey, endpoint: endpoint, http: h}
}

var _ ports.Provider = (*Client)(nil)

func (c *Client) ID() domain.ProviderID { return domain.ProviderAssemblyAI }
func (c *Client) MaxUploadBytes() int64 { return maxUploadBytes }
func (c *Client) Models() []string      { return Models() }
func (c *Client) DefaultModel() string  { return DefaultModel() }
func (c *Client) Capabilities(m string) ports.ModelCapabilities {
	return Capabilities(m)
}

// CheckKey verifies the API key by listing one transcript (free, non-consuming).
// AssemblyAI has no dedicated /user endpoint; transcript listing is the cheapest
// authenticated GET. The Python client skipped validation entirely.
func (c *Client) CheckKey(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, checkKeyTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", c.endpoint+checkKeyPath, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", c.apiKey) // raw key, no Bearer prefix
	resp, err := c.http.Do(req)
	if err != nil {
		return &domain.ErrProvider{Provider: domain.ProviderAssemblyAI, Cause: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return &domain.ErrProvider{
			Provider:   domain.ProviderAssemblyAI,
			StatusCode: resp.StatusCode,
			Cause:      fmt.Errorf("%s", string(body)),
		}
	}
	return nil
}

func (c *Client) Transcribe(ctx context.Context, audio domain.AudioFile, opts ports.ProviderOpts) (*domain.Result, error) {
	model := opts.Model
	if model == "" {
		model = DefaultModel()
	}

	// Step 1: Upload raw audio bytes.
	uploadURL, err := c.uploadFile(ctx, audio.Path)
	if err != nil {
		return nil, &domain.ErrProvider{
			Provider:  domain.ProviderAssemblyAI,
			Retryable: retry.IsRetryable(err),
			Cause:     fmt.Errorf("upload: %w", err),
		}
	}

	// Step 2: Submit transcript request.
	transcriptID, err := c.submitTranscript(ctx, uploadURL, model, opts)
	if err != nil {
		return nil, &domain.ErrProvider{
			Provider:  domain.ProviderAssemblyAI,
			Retryable: retry.IsRetryable(err),
			Cause:     fmt.Errorf("submit: %w", err),
		}
	}

	// Step 3: Poll until completed or error.
	var raw []byte
	err = retry.Do(ctx, 3, 5*time.Second, func() error {
		var pollErr error
		raw, pollErr = c.pollUntilDone(ctx, transcriptID)
		return pollErr
	})
	if err != nil {
		return nil, &domain.ErrProvider{
			Provider:  domain.ProviderAssemblyAI,
			Retryable: retry.IsRetryable(err),
			Cause:     fmt.Errorf("poll: %w", err),
		}
	}
	return parse(raw, model)
}

// uploadFile sends the audio file as raw bytes to /v2/upload and returns the upload URL.
func (c *Client) uploadFile(ctx context.Context, path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	var uploadURL string
	err = retry.Do(ctx, 3, 5*time.Second, func() error {
		// Seek to start on retry
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return err
		}
		req, err := http.NewRequestWithContext(ctx, "POST", c.endpoint+uploadPath, f)
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", c.apiKey)
		req.Header.Set("Content-Type", "application/octet-stream")

		resp, err := c.http.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		if resp.StatusCode/100 != 2 {
			return retry.HTTPError{StatusCode: resp.StatusCode, Message: string(data)}
		}
		var result struct {
			UploadURL string `json:"upload_url"`
		}
		if err := json.Unmarshal(data, &result); err != nil {
			return err
		}
		uploadURL = result.UploadURL
		return nil
	})
	return uploadURL, err
}

// buildSpeechModels assembles AssemblyAI's ordered `speech_models` fallback
// array. It starts from the explicit fallback list (if any) or the single
// selected model, always ensures fallbackModel is the final entry, and dedupes
// while preserving order. AssemblyAI's official param is the plural array; the
// singular `speech_model` is no longer sent.
func buildSpeechModels(primary string, fallbacks []string) []string {
	src := fallbacks
	if len(src) == 0 {
		src = []string{primary}
	}
	src = append(append([]string{}, src...), fallbackModel)
	seen := make(map[string]bool, len(src))
	out := make([]string, 0, len(src))
	for _, m := range src {
		if m == "" || seen[m] {
			continue
		}
		seen[m] = true
		out = append(out, m)
	}
	return out
}

// submitTranscript posts a transcript request and returns the transcript ID.
func (c *Client) submitTranscript(ctx context.Context, audioURL, model string, opts ports.ProviderOpts) (string, error) {
	body := map[string]interface{}{
		"audio_url":     audioURL,
		"speech_models": buildSpeechModels(model, opts.SpeechModels),
		"language_code": opts.Language,
	}
	if opts.Language == "" {
		delete(body, "language_code")
		body["language_detection"] = true
	}
	if opts.SpeakerLabels {
		body["speaker_labels"] = true
		if opts.NumSpeakers > 0 {
			body["speakers_expected"] = opts.NumSpeakers
		}
	}
	if len(opts.KeyTerms) > 0 {
		body["keyterms_prompt"] = opts.KeyTerms
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return "", err
	}

	var transcriptID string
	err = retry.Do(ctx, 3, 5*time.Second, func() error {
		req, err := http.NewRequestWithContext(ctx, "POST", c.endpoint+transcriptPath, bytes.NewReader(payload))
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", c.apiKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := c.http.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		if resp.StatusCode/100 != 2 {
			return retry.HTTPError{StatusCode: resp.StatusCode, Message: string(data)}
		}
		var result struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(data, &result); err != nil {
			return err
		}
		transcriptID = result.ID
		return nil
	})
	return transcriptID, err
}

// pollUntilDone polls GET /v2/transcript/{id} until status is "completed" or "error".
func (c *Client) pollUntilDone(ctx context.Context, transcriptID string) ([]byte, error) {
	pollURL := c.endpoint + transcriptPath + "/" + transcriptID
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		req, err := http.NewRequestWithContext(ctx, "GET", pollURL, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", c.apiKey)

		resp, err := c.http.Do(req)
		if err != nil {
			return nil, err
		}
		data, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
		if resp.StatusCode/100 != 2 {
			return nil, retry.HTTPError{StatusCode: resp.StatusCode, Message: string(data)}
		}

		var status struct {
			Status string `json:"status"`
		}
		if err := json.Unmarshal(data, &status); err != nil {
			return nil, err
		}

		switch status.Status {
		case "completed", "error":
			return data, nil
		}

		// Still processing — wait and retry.
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(pollInterval):
		}
	}
}
