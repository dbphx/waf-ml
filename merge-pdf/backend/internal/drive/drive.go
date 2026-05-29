package drive

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/ml/merge-pdf/backend/internal/model"
)

var folderPatterns = []*regexp.Regexp{
	regexp.MustCompile(`/folders/([a-zA-Z0-9_-]+)`),
	regexp.MustCompile(`id=([a-zA-Z0-9_-]+)`),
}

var orderPattern = regexp.MustCompile(`(\d+)`)

type Client struct {
	apiKey     string
	httpClient *http.Client
}

// NewClient wraps shared Drive HTTP configuration for preview and download requests.
func NewClient(apiKey string, httpClient *http.Client) Client {
	return Client{apiKey: apiKey, httpClient: httpClient}
}

// ExtractFolderID normalizes shared Drive folder links into the folder identifier used by the API.
func ExtractFolderID(rawURL string) (string, error) {
	for _, pattern := range folderPatterns {
		matches := pattern.FindStringSubmatch(rawURL)
		if len(matches) == 2 {
			return matches[1], nil
		}
	}
	return "", fmt.Errorf("could not extract Google Drive folder id")
}

// ExtractOrder enforces the filename-based ordering rule agreed for Drive merges.
func ExtractOrder(filename string) (int, error) {
	match := orderPattern.FindStringSubmatch(filepath.Base(filename))
	if len(match) != 2 {
		return 0, fmt.Errorf("filename %q does not contain an ordering number", filename)
	}

	order, err := strconv.Atoi(match[1])
	if err != nil {
		return 0, fmt.Errorf("parse order from %q: %w", filename, err)
	}
	return order, nil
}

type driveListResponse struct {
	Files []struct {
		ID             string `json:"id"`
		Name           string `json:"name"`
		MimeType       string `json:"mimeType"`
		Size           string `json:"size"`
		WebViewLink    string `json:"webViewLink"`
		WebContentLink string `json:"webContentLink"`
	} `json:"files"`
}

// PreviewFolder lists direct PDF children for a shared Drive folder and pre-sorts them for merge.
func (c Client) PreviewFolder(ctx context.Context, folderURL string) ([]model.DrivePreviewFile, error) {
	if c.apiKey == "" {
		return nil, fmt.Errorf("GOOGLE_DRIVE_API_KEY is required for Drive preview")
	}

	folderID, err := ExtractFolderID(folderURL)
	if err != nil {
		return nil, err
	}

	query := url.Values{}
	query.Set("q", fmt.Sprintf("'%s' in parents and trashed=false", folderID))
	query.Set("fields", "files(id,name,mimeType,size,webViewLink,webContentLink)")
	query.Set("key", c.apiKey)
	query.Set("includeItemsFromAllDrives", "true")
	query.Set("supportsAllDrives", "true")

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://www.googleapis.com/drive/v3/files?"+query.Encode(), nil)
	if err != nil {
		return nil, fmt.Errorf("build drive list request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list drive files: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("drive preview failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var payload driveListResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode drive response: %w", err)
	}

	files := make([]model.DrivePreviewFile, 0, len(payload.Files))
	for _, file := range payload.Files {
		if file.MimeType != "application/pdf" {
			continue
		}
		order, err := ExtractOrder(file.Name)
		if err != nil {
			return nil, err
		}
		size, _ := strconv.ParseInt(file.Size, 10, 64)
		files = append(files, model.DrivePreviewFile{
			SourceID:       file.ID,
			Name:           file.Name,
			Size:           size,
			ExtractedOrder: order,
			WebViewLink:    file.WebViewLink,
		})
	}

	if len(files) == 0 {
		return nil, fmt.Errorf("no PDF files found in the provided folder")
	}

	sort.Slice(files, func(i, j int) bool {
		if files[i].ExtractedOrder == files[j].ExtractedOrder {
			return strings.ToLower(files[i].Name) < strings.ToLower(files[j].Name)
		}
		return files[i].ExtractedOrder < files[j].ExtractedOrder
	})

	return files, nil
}

// DownloadFile streams a shared Drive PDF into the merge workspace without persisting source files.
func (c Client) DownloadFile(ctx context.Context, fileID string) (io.ReadCloser, error) {
	if c.apiKey == "" {
		return nil, fmt.Errorf("GOOGLE_DRIVE_API_KEY is required for Drive download")
	}

	downloadURL := fmt.Sprintf("https://www.googleapis.com/drive/v3/files/%s?alt=media&key=%s", url.PathEscape(fileID), url.QueryEscape(c.apiKey))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, downloadURL, nil)
	if err != nil {
		return nil, fmt.Errorf("build drive download request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download drive file: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("drive download failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	return resp.Body, nil
}
