package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/ml/merge-pdf/backend/internal/auth"
	"github.com/ml/merge-pdf/backend/internal/config"
	"github.com/ml/merge-pdf/backend/internal/drive"
	"github.com/ml/merge-pdf/backend/internal/merge"
	"github.com/ml/merge-pdf/backend/internal/model"
	"github.com/ml/merge-pdf/backend/internal/repository"
	"github.com/ml/merge-pdf/backend/internal/storage"
)

type Server struct {
	cfg           config.Config
	repo          *repository.Repository
	auth          auth.Service
	drive         drive.Client
	storage       *storage.Client
	httpServer    *http.Server
	maxUploadMB   int64
	allowedOrigin string
}

type authContextKey struct{}

type loginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

type drivePreviewRequest struct {
	URL string `json:"url"`
}

type driveMergeRequest struct {
	URL string `json:"url"`
}

// New wires the API surface once so every request path shares the same auth, storage, and timeout policy.
func New(cfg config.Config, repo *repository.Repository, authSvc auth.Service, driveClient drive.Client, storageClient *storage.Client) *Server {
	s := &Server{
		cfg:           cfg,
		repo:          repo,
		auth:          authSvc,
		drive:         driveClient,
		storage:       storageClient,
		maxUploadMB:   cfg.MaxUploadBytes,
		allowedOrigin: cfg.AllowedOrigin,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api/auth/login", s.handleLogin)
	mux.HandleFunc("/api/auth/logout", s.handleLogout)
	mux.Handle("/api/me", s.withAuth(http.HandlerFunc(s.handleMe)))
	mux.Handle("/api/drive/preview", s.withAuth(http.HandlerFunc(s.handleDrivePreview)))
	mux.Handle("/api/merge/drive", s.withAuth(http.HandlerFunc(s.handleDriveMerge)))
	mux.Handle("/api/merge/upload", s.withAuth(http.HandlerFunc(s.handleUploadMerge)))
	mux.Handle("/api/jobs", s.withAuth(http.HandlerFunc(s.handleJobs)))
	mux.Handle("/api/jobs/", s.withAuth(http.HandlerFunc(s.handleJobByID)))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	s.httpServer = &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      s.withCORS(s.withLogging(mux)),
		ReadTimeout:  cfg.RequestTimeout,
		WriteTimeout: cfg.RequestTimeout,
		IdleTimeout:  30 * time.Second,
	}

	return s
}

// Start owns the HTTP listener lifecycle for local dev and deployed environments.
func (s *Server) Start() error {
	log.Printf("listening on %s", s.httpServer.Addr)
	return s.httpServer.ListenAndServe()
}

// Shutdown gives in-flight merge requests a brief drain window during process termination.
func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

func (s *Server) withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}

func (s *Server) withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", s.allowedOrigin)
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) withAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header := r.Header.Get("Authorization")
		if !strings.HasPrefix(header, "Bearer ") {
			writeError(w, http.StatusUnauthorized, "missing bearer token")
			return
		}

		claims, err := s.auth.ParseToken(strings.TrimPrefix(header, "Bearer "))
		if err != nil {
			writeError(w, http.StatusUnauthorized, "invalid token")
			return
		}

		user, err := s.repo.GetUserByID(r.Context(), claims.UserID)
		if err != nil {
			writeError(w, http.StatusUnauthorized, "user not found")
			return
		}

		ctx := context.WithValue(r.Context(), authContextKey{}, user)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req loginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid login payload")
		return
	}

	user, err := s.repo.GetUserByEmail(r.Context(), strings.ToLower(strings.TrimSpace(req.Email)))
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			writeError(w, http.StatusUnauthorized, "invalid credentials")
			return
		}
		writeError(w, http.StatusInternalServerError, "failed to load user")
		return
	}

	if err := s.auth.CheckPassword(user.PasswordHash, req.Password); err != nil {
		writeError(w, http.StatusUnauthorized, "invalid credentials")
		return
	}

	token, err := s.auth.GenerateToken(user)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create token")
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"token": token,
		"user": map[string]any{
			"id":    user.ID,
			"email": user.Email,
			"role":  user.Role,
		},
	})
}

func (s *Server) handleLogout(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleMe(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	user := currentUser(r.Context())
	writeJSON(w, http.StatusOK, map[string]any{
		"id":    user.ID,
		"email": user.Email,
		"role":  user.Role,
	})
}

func (s *Server) handleDrivePreview(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req drivePreviewRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid drive preview payload")
		return
	}

	files, err := s.drive.PreviewFolder(r.Context(), req.URL)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{"files": files})
}

func (s *Server) handleDriveMerge(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req driveMergeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid drive merge payload")
		return
	}

	files, err := s.drive.PreviewFolder(r.Context(), req.URL)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	jobFiles := make([]model.JobFile, 0, len(files))
	for _, file := range files {
		sizeCopy := file.Size
		sourceID := file.SourceID
		driveLink := file.WebViewLink
		jobFiles = append(jobFiles, model.JobFile{
			SourceKind:  string(model.SourceTypeDrive),
			SourceName:  file.Name,
			SourceOrder: file.ExtractedOrder,
			SourceSize:  &sizeCopy,
			DriveFileID: &sourceID,
			DriveLink:   &driveLink,
		})
	}

	job, err := s.repo.CreateJob(r.Context(), currentUser(r.Context()).ID, model.SourceTypeDrive, model.JobStatusPending, 5, "drive-merged.pdf", jobFiles)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create job")
		return
	}

	go s.processDriveMerge(job.ID, files)
	writeJSON(w, http.StatusAccepted, job)
}

func (s *Server) processDriveMerge(jobID int64, files []model.DrivePreviewFile) {
	workDir, err := os.MkdirTemp("", "mergepdf-drive-*")
	if err != nil {
		s.failJob(jobID, 5, "failed to create work dir")
		return
	}
	defer os.RemoveAll(workDir)
	inputs := make([]model.MergeFileInput, 0, len(files))
	ctx := context.Background()
	_ = s.repo.UpdateJobProgress(ctx, jobID, model.JobStatusRunning, 10)
	for index, file := range files {
		reader, err := s.drive.DownloadFile(ctx, file.SourceID)
		if err != nil {
			s.failJob(jobID, progressStep(index, len(files), 10, 70), err.Error())
			return
		}

		localPath := filepath.Join(workDir, file.Name)
		size, err := saveUploadedReader(localPath, reader)
		reader.Close()
		if err != nil {
			s.failJob(jobID, progressStep(index, len(files), 10, 70), "failed to save drive file")
			return
		}

		inputs = append(inputs, model.MergeFileInput{
			Name:      file.Name,
			LocalPath: localPath,
			Order:     file.ExtractedOrder,
			Size:      size,
			SourceID:  file.SourceID,
			DriveLink: file.WebViewLink,
		})
		_ = s.repo.UpdateJobProgress(ctx, jobID, model.JobStatusRunning, progressStep(index+1, len(files), 10, 70))
	}

	s.finishMergeJob(ctx, jobID, workDir, "drive-merged.pdf", inputs)
}

func (s *Server) handleUploadMerge(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, s.maxUploadMB)
	if err := r.ParseMultipartForm(s.maxUploadMB); err != nil {
		writeError(w, http.StatusBadRequest, "failed to parse upload form")
		return
	}

	orderPayload := r.FormValue("orders")
	var orders map[string]int
	if err := json.Unmarshal([]byte(orderPayload), &orders); err != nil {
		writeError(w, http.StatusBadRequest, "invalid orders payload")
		return
	}

	workDir, err := os.MkdirTemp("", "mergepdf-upload-*")
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create work dir")
		return
	}
	shouldCleanupWorkDir := true
	defer func() {
		if shouldCleanupWorkDir {
			os.RemoveAll(workDir)
		}
	}()

	multipartFiles := r.MultipartForm.File["files"]
	inputs := make([]model.MergeFileInput, 0, len(multipartFiles))
	jobFiles := make([]model.JobFile, 0, len(multipartFiles))
	for _, header := range multipartFiles {
		order, ok := orders[header.Filename]
		if !ok {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("missing order for %s", header.Filename))
			return
		}
		if strings.ToLower(filepath.Ext(header.Filename)) != ".pdf" {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("%s is not a PDF", header.Filename))
			return
		}

		localPath, size, err := saveMultipartFile(workDir, header)
		if err != nil {
			writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to save %s", header.Filename))
			return
		}

		inputs = append(inputs, model.MergeFileInput{
			Name:      header.Filename,
			LocalPath: localPath,
			Order:     order,
			Size:      size,
		})

		sizeCopy := size
		jobFiles = append(jobFiles, model.JobFile{
			SourceKind:  string(model.SourceTypeUpload),
			SourceName:  header.Filename,
			SourceOrder: order,
			SourceSize:  &sizeCopy,
		})
	}

	job, err := s.repo.CreateJob(r.Context(), currentUser(r.Context()).ID, model.SourceTypeUpload, model.JobStatusPending, 25, "upload-merged.pdf", jobFiles)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to create job")
		return
	}

	shouldCleanupWorkDir = false
	go s.processUploadMerge(job.ID, workDir, inputs)
	writeJSON(w, http.StatusAccepted, job)
}

func (s *Server) processUploadMerge(jobID int64, workDir string, inputs []model.MergeFileInput) {
	defer os.RemoveAll(workDir)
	ctx := context.Background()
	_ = s.repo.UpdateJobProgress(ctx, jobID, model.JobStatusRunning, 40)
	s.finishMergeJob(ctx, jobID, workDir, "upload-merged.pdf", inputs)
}

func (s *Server) finishMergeJob(ctx context.Context, jobID int64, workDir, outputName string, inputs []model.MergeFileInput) {
	_ = s.repo.UpdateJobProgress(ctx, jobID, model.JobStatusRunning, 75)
	outputPath, err := merge.MergeFiles(workDir, outputName, inputs)
	if err != nil {
		s.failJob(jobID, 75, err.Error())
		return
	}

	file, err := os.Open(outputPath)
	if err != nil {
		s.failJob(jobID, 80, "failed to open merged output")
		return
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		s.failJob(jobID, 80, "failed to stat merged output")
		return
	}

	job, err := s.repo.GetJob(ctx, jobID)
	if err != nil {
		s.failJob(jobID, 80, "failed to reload job")
		return
	}

	objectKey := fmt.Sprintf("jobs/%d/%d-%s", job.UserID, time.Now().UnixNano(), outputName)
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		s.failJob(jobID, 85, "failed to rewind merged output")
		return
	}
	_ = s.repo.UpdateJobProgress(ctx, jobID, model.JobStatusRunning, 90)
	if err := s.storage.Upload(ctx, objectKey, file, info.Size()); err != nil {
		s.failJob(jobID, 90, "failed to upload merged output")
		return
	}
	if err := s.repo.CompleteJob(ctx, jobID, objectKey); err != nil {
		s.failJob(jobID, 95, "failed to finalize job history")
	}
}

func (s *Server) handleJobs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	jobs, err := s.repo.ListJobs(r.Context(), currentUser(r.Context()))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to list jobs")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"jobs": jobs})
}

func (s *Server) handleJobByID(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/api/jobs/")
	if path == "" {
		writeError(w, http.StatusNotFound, "job route not found")
		return
	}

	if strings.HasSuffix(path, "/download") {
		idValue := strings.TrimSuffix(path, "/download")
		jobID, err := strconv.ParseInt(strings.TrimSuffix(idValue, "/"), 10, 64)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid job id")
			return
		}
		s.handleJobDownload(w, r, jobID)
		return
	}

	jobID, err := strconv.ParseInt(strings.TrimSuffix(path, "/"), 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid job id")
		return
	}

	switch r.Method {
	case http.MethodGet:
		s.handleJobDetail(w, r, jobID)
	case http.MethodDelete:
		s.handleJobDelete(w, r, jobID)
	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (s *Server) handleJobDetail(w http.ResponseWriter, r *http.Request, jobID int64) {
	job, err := s.repo.GetJob(r.Context(), jobID)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			writeError(w, http.StatusNotFound, "job not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "failed to load job")
		return
	}
	if !auth.CanAccessJob(currentUser(r.Context()), job.UserID) {
		writeError(w, http.StatusForbidden, "forbidden")
		return
	}

	writeJSON(w, http.StatusOK, job)
}

func (s *Server) handleJobDownload(w http.ResponseWriter, r *http.Request, jobID int64) {
	job, err := s.repo.GetJob(r.Context(), jobID)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			writeError(w, http.StatusNotFound, "job not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "failed to load job")
		return
	}
	if !auth.CanAccessJob(currentUser(r.Context()), job.UserID) {
		writeError(w, http.StatusForbidden, "forbidden")
		return
	}
	if job.Status != model.JobStatusCompleted || job.OutputObjectKey == "" {
		writeError(w, http.StatusConflict, "job is not ready for download")
		return
	}

	object, err := s.storage.Download(r.Context(), job.OutputObjectKey)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to load merged file")
		return
	}
	defer object.Close()

	stat, err := object.Stat()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to stat merged file")
		return
	}

	w.Header().Set("Content-Type", "application/pdf")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", job.OutputFilename))
	w.Header().Set("Content-Length", strconv.FormatInt(stat.Size, 10))
	io.Copy(w, object)
}

func (s *Server) handleJobDelete(w http.ResponseWriter, r *http.Request, jobID int64) {
	job, err := s.repo.GetJob(r.Context(), jobID)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			writeError(w, http.StatusNotFound, "job not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "failed to load job")
		return
	}
	if !auth.CanAccessJob(currentUser(r.Context()), job.UserID) {
		writeError(w, http.StatusForbidden, "forbidden")
		return
	}

	if err := s.storage.Delete(r.Context(), job.OutputObjectKey); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to delete merged file")
		return
	}
	if err := s.repo.DeleteJob(r.Context(), jobID); err != nil {
		writeError(w, http.StatusInternalServerError, "failed to delete job")
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}

func saveMultipartFile(workDir string, header *multipart.FileHeader) (string, int64, error) {
	src, err := header.Open()
	if err != nil {
		return "", 0, err
	}
	defer src.Close()

	dstPath := filepath.Join(workDir, header.Filename)
	size, err := saveUploadedReader(dstPath, src)
	if err != nil {
		return "", 0, err
	}
	return dstPath, size, nil
}

func (s *Server) failJob(jobID int64, progressPercent int, message string) {
	if err := s.repo.FailJob(context.Background(), jobID, progressPercent, message); err != nil {
		log.Printf("failed to mark job %d as failed: %v", jobID, err)
	}
}

func progressStep(current, total, min, max int) int {
	if total <= 0 {
		return max
	}
	if current < 0 {
		current = 0
	}
	if current > total {
		current = total
	}
	return min + ((max - min) * current / total)
}

func saveUploadedReader(dstPath string, src io.Reader) (int64, error) {
	dst, err := os.Create(dstPath)
	if err != nil {
		return 0, err
	}
	defer dst.Close()

	size, err := io.Copy(dst, src)
	if err != nil {
		return 0, err
	}
	return size, nil
}

func currentUser(ctx context.Context) model.User {
	return ctx.Value(authContextKey{}).(model.User)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}
