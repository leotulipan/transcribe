package groq

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/leotulipan/transcribe/internal/adapters/api/internal/retry"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

const (
	defaultEndpoint = "https://api.groq.com"
	maxUploadBytes  = 25 * 1024 * 1024
	transcribePath  = "/openai/v1/audio/transcriptions"
	modelsPath      = "/openai/v1/models"
	requestTimeout  = 5 * time.Minute
	checkKeyTimeout = 10 * time.Second
)

// Client implements ports.Provider against Groq's OpenAI-compatible endpoint.
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

func (c *Client) ID() domain.ProviderID { return domain.ProviderGroq }
func (c *Client) MaxUploadBytes() int64 { return maxUploadBytes }
func (c *Client) Models() []string      { return Models() }
func (c *Client) DefaultModel() string  { return DefaultModel() }
func (c *Client) Capabilities(m string) ports.ModelCapabilities {
	return Capabilities(m)
}

// CheckKey verifies the API key with a non-consuming GET /openai/v1/models.
// Listing models is free and a 200 response confirms the key is accepted.
func (c *Client) CheckKey(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, checkKeyTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", c.endpoint+modelsPath, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	resp, err := c.http.Do(req)
	if err != nil {
		return &domain.ErrProvider{Provider: domain.ProviderGroq, Cause: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return &domain.ErrProvider{
			Provider:   domain.ProviderGroq,
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
	var raw []byte
	err := retry.Do(ctx, 3, 5*time.Second, func() error {
		body, contentType, err := buildMultipart(audio.Path, model, opts.Language)
		if err != nil {
			return err
		}
		req, err := http.NewRequestWithContext(ctx, "POST", c.endpoint+transcribePath, body)
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
		req.Header.Set("Content-Type", contentType)

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
		raw = data
		return nil
	})
	if err != nil {
		return nil, &domain.ErrProvider{
			Provider:  domain.ProviderGroq,
			Retryable: retry.IsRetryable(err),
			Cause:     err,
		}
	}
	return parse(raw, model)
}

func buildMultipart(path, model, language string) (*bytes.Buffer, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	buf := &bytes.Buffer{}
	mw := multipart.NewWriter(buf)
	if err := mw.WriteField("model", model); err != nil {
		return nil, "", err
	}
	if err := mw.WriteField("response_format", "verbose_json"); err != nil {
		return nil, "", err
	}
	if err := mw.WriteField("timestamp_granularities[]", "word"); err != nil {
		return nil, "", err
	}
	if err := mw.WriteField("timestamp_granularities[]", "segment"); err != nil {
		return nil, "", err
	}
	if language != "" {
		if err := mw.WriteField("language", language); err != nil {
			return nil, "", err
		}
	}
	fw, err := mw.CreateFormFile("file", filepath.Base(path))
	if err != nil {
		return nil, "", err
	}
	if _, err := io.Copy(fw, f); err != nil {
		return nil, "", err
	}
	if err := mw.Close(); err != nil {
		return nil, "", err
	}
	return buf, mw.FormDataContentType(), nil
}
