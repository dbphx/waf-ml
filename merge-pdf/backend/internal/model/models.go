package model

import "time"

type Role string

const (
	RoleAdmin Role = "admin"
	RoleUser  Role = "user"
)

type User struct {
	ID           int64     `json:"id"`
	Email        string    `json:"email"`
	PasswordHash string    `json:"-"`
	Role         Role      `json:"role"`
	CreatedAt    time.Time `json:"createdAt"`
}

type SourceType string

const (
	SourceTypeDrive  SourceType = "drive"
	SourceTypeUpload SourceType = "upload"
)

type JobStatus string

const (
	JobStatusPending   JobStatus = "pending"
	JobStatusRunning   JobStatus = "running"
	JobStatusCompleted JobStatus = "completed"
	JobStatusFailed    JobStatus = "failed"
)

type Job struct {
	ID              int64      `json:"id"`
	UserID          int64      `json:"userId"`
	SourceType      SourceType `json:"sourceType"`
	Status          JobStatus  `json:"status"`
	OutputObjectKey string     `json:"-"`
	OutputFilename  string     `json:"outputFilename"`
	ProgressPercent int        `json:"progressPercent"`
	ErrorMessage    *string    `json:"errorMessage,omitempty"`
	CreatedAt       time.Time  `json:"createdAt"`
	Files           []JobFile  `json:"files,omitempty"`
}

type JobFile struct {
	ID          int64   `json:"id"`
	JobID       int64   `json:"jobId"`
	SourceKind  string  `json:"sourceKind"`
	SourceName  string  `json:"name"`
	SourceOrder int     `json:"order"`
	SourceSize  *int64  `json:"size,omitempty"`
	DriveFileID *string `json:"driveFileId,omitempty"`
	DriveLink   *string `json:"driveLink,omitempty"`
}

type DrivePreviewFile struct {
	SourceID       string `json:"sourceId"`
	Name           string `json:"name"`
	Size           int64  `json:"size"`
	ExtractedOrder int    `json:"extractedOrder"`
	WebViewLink    string `json:"webViewLink"`
}

type MergeFileInput struct {
	Name      string
	LocalPath string
	Order     int
	Size      int64
	SourceID  string
	DriveLink string
}
