package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/leotulipan/transcribe/internal/adapters/api/internal/retry"
	"github.com/leotulipan/transcribe/internal/core/domain"
	"github.com/leotulipan/transcribe/internal/ports"
)

const (
	defaultEndpoint = "https://generativelanguage.googleapis.com"
	maxUploadBytes  = 2 * 1024 * 1024 * 1024 // 2 GB (Files API limit)
	uploadBasePath  = "/upload/v1beta/files"
	generatePath    = "/v1beta/models/%s:generateContent"
	modelsPath      = "/v1beta/models"
	requestTimeout  = 10 * time.Minute
	checkKeyTimeout = 10 * time.Second
	transcribePrompt = "Transcribe the audio. Return only the spoken text, with no commentary."
)

// Client implements ports.Provider against Google Gemini's Files API + generateContent.
// Two-step flow:
//  1. Upload audio to Files API → get file URI
//  2. Call generateContent with file_data referencing the URI
//
// Auth: x-goog-api-key header.
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

func (c *Client) ID() domain.ProviderID { return domain.ProviderGemini }
func (c *Client) MaxUploadBytes() int64 { return maxUploadBytes }
func (c *Client) Models() []string      { return Models() }
func (c *Client) DefaultModel() string  { return DefaultModel() }
func (c *Client) Capabilities(m string) ports.ModelCapabilities {
	return Capabilities(m)
}

// CheckKey verifies the API key with a non-consuming GET /v1beta/models.
func (c *Client) CheckKey(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, checkKeyTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", c.endpoint+modelsPath, nil)
	if err != nil {
		return err
	}
	req.Header.Set("x-goog-api-key", c.apiKey)
	resp, err := c.http.Do(req)
	if err != nil {
		return &domain.ErrProvider{Provider: domain.ProviderGemini, Cause: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return &domain.ErrProvider{
			Provider:   domain.ProviderGemini,
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

	// Step 1: Upload the file to the Files API.
	mimeType := mimeTypeForCodec(audio.Codec)
	fileURI, err := c.uploadFile(ctx, audio.Path, mimeType)
	if err != nil {
		return nil, &domain.ErrProvider{
			Provider:  domain.ProviderGemini,
			Retryable: retry.IsRetryable(err),
			Cause:     fmt.Errorf("upload: %w", err),
		}
	}

	// Step 2: Generate content using the uploaded file.
	var raw []byte
	err = retry.Do(ctx, 3, 5*time.Second, func() error {
		var genErr error
		raw, genErr = c.generateContent(ctx, model, fileURI, mimeType)
		return genErr
	})
	if err != nil {
		return nil, &domain.ErrProvider{
			Provider:  domain.ProviderGemini,
			Retryable: retry.IsRetryable(err),
			Cause:     fmt.Errorf("generateContent: %w", err),
		}
	}
	return parse(raw, model)
}

// uploadFile uploads the audio file using Gemini's Files API resumable protocol.
// Returns the file URI for use in generateContent.
func (c *Client) uploadFile(ctx context.Context, path, mimeType string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	displayName := filepath.Base(path)

	var fileURI string
	err = retry.Do(ctx, 3, 5*time.Second, func() error {
		// Use the multipart upload approach: POST with metadata + data in a single request.
		// Gemini Files API supports X-Goog-Upload-Protocol: multipart.
		metaJSON, _ := json.Marshal(map[string]interface{}{
			"file": map[string]string{"display_name": displayName},
		})
		var buf bytes.Buffer
		boundary := "gemini_upload_boundary"
		buf.WriteString("--" + boundary + "\r\n")
		buf.WriteString("Content-Type: application/json; charset=UTF-8\r\n\r\n")
		buf.Write(metaJSON)
		buf.WriteString("\r\n--" + boundary + "\r\n")
		buf.WriteString("Content-Type: " + mimeType + "\r\n\r\n")
		buf.Write(data)
		buf.WriteString("\r\n--" + boundary + "--\r\n")

		req2, err := http.NewRequestWithContext(ctx, "POST", c.endpoint+uploadBasePath, &buf)
		if err != nil {
			return err
		}
		req2.Header.Set("x-goog-api-key", c.apiKey)
		req2.Header.Set("X-Goog-Upload-Protocol", "multipart")
		req2.Header.Set("Content-Type", "multipart/related; boundary="+boundary)

		resp, err := c.http.Do(req2)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		respData, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		if resp.StatusCode/100 != 2 {
			return retry.HTTPError{StatusCode: resp.StatusCode, Message: string(respData)}
		}
		var result struct {
			File struct {
				URI string `json:"uri"`
			} `json:"file"`
		}
		if err := json.Unmarshal(respData, &result); err != nil {
			return err
		}
		if result.File.URI == "" {
			return fmt.Errorf("gemini: upload returned empty file URI")
		}
		fileURI = result.File.URI
		return nil
	})
	return fileURI, err
}

// generateContent calls the Gemini generateContent endpoint with the uploaded file URI.
func (c *Client) generateContent(ctx context.Context, model, fileURI, mimeType string) ([]byte, error) {
	body := map[string]interface{}{
		"contents": []interface{}{
			map[string]interface{}{
				"parts": []interface{}{
					map[string]string{"text": transcribePrompt},
					map[string]interface{}{
						"file_data": map[string]string{
							"mime_type": mimeType,
							"file_uri":  fileURI,
						},
					},
				},
			},
		},
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	url := c.endpoint + fmt.Sprintf(generatePath, model)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-goog-api-key", c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		return nil, retry.HTTPError{StatusCode: resp.StatusCode, Message: string(data)}
	}
	return data, nil
}

// mimeTypeForCodec returns the MIME type for a given audio codec/container.
func mimeTypeForCodec(codec string) string {
	switch codec {
	case "mp3", "mpeg":
		return "audio/mpeg"
	case "wav":
		return "audio/wav"
	case "aiff":
		return "audio/aiff"
	case "aac":
		return "audio/aac"
	case "ogg":
		return "audio/ogg"
	case "flac":
		return "audio/flac"
	case "mp4", "m4a":
		return "audio/mp4"
	case "webm":
		return "audio/webm"
	default:
		// Fall back to a generic type and let Gemini figure it out.
		t := mime.TypeByExtension("." + codec)
		if t == "" {
			return "application/octet-stream"
		}
		return t
	}
}
