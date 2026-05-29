package repository

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ml/merge-pdf/backend/internal/model"
)

type Repository struct {
	db *pgxpool.Pool
}

// New binds the repository to a connection pool so handlers can share one database access layer.
func New(db *pgxpool.Pool) *Repository {
	return &Repository{db: db}
}

// GetUserByEmail powers email-password login lookups against the canonical user record.
func (r *Repository) GetUserByEmail(ctx context.Context, email string) (model.User, error) {
	const query = `
		SELECT id, email, password_hash, role, created_at
		FROM users
		WHERE email = $1
	`

	var user model.User
	err := r.db.QueryRow(ctx, query, email).Scan(&user.ID, &user.Email, &user.PasswordHash, &user.Role, &user.CreatedAt)
	if err != nil {
		return model.User{}, err
	}
	return user, nil
}

// GetUserByID reloads the current user on each request so authorization uses fresh role data.
func (r *Repository) GetUserByID(ctx context.Context, id int64) (model.User, error) {
	const query = `
		SELECT id, email, password_hash, role, created_at
		FROM users
		WHERE id = $1
	`

	var user model.User
	err := r.db.QueryRow(ctx, query, id).Scan(&user.ID, &user.Email, &user.PasswordHash, &user.Role, &user.CreatedAt)
	if err != nil {
		return model.User{}, err
	}
	return user, nil
}

// CreateJob records both the merged output and its source manifest in one transaction for history integrity.
func (r *Repository) CreateJob(ctx context.Context, userID int64, sourceType model.SourceType, outputFilename, outputObjectKey string, files []model.JobFile) (model.Job, error) {
	tx, err := r.db.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return model.Job{}, fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	const insertJob = `
		INSERT INTO jobs (user_id, source_type, status, output_filename, output_object_key)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, user_id, source_type, status, output_filename, output_object_key, created_at
	`

	var job model.Job
	err = tx.QueryRow(ctx, insertJob, userID, sourceType, model.JobStatusCompleted, outputFilename, outputObjectKey).
		Scan(&job.ID, &job.UserID, &job.SourceType, &job.Status, &job.OutputFilename, &job.OutputObjectKey, &job.CreatedAt)
	if err != nil {
		return model.Job{}, fmt.Errorf("insert job: %w", err)
	}

	const insertFile = `
		INSERT INTO job_files (job_id, source_kind, source_name, source_order, source_size, drive_file_id, drive_link)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		RETURNING id, job_id, source_kind, source_name, source_order, source_size, drive_file_id, drive_link
	`

	job.Files = make([]model.JobFile, 0, len(files))
	for _, file := range files {
		var saved model.JobFile
		err = tx.QueryRow(ctx, insertFile, job.ID, file.SourceKind, file.SourceName, file.SourceOrder, file.SourceSize, file.DriveFileID, file.DriveLink).
			Scan(&saved.ID, &saved.JobID, &saved.SourceKind, &saved.SourceName, &saved.SourceOrder, &saved.SourceSize, &saved.DriveFileID, &saved.DriveLink)
		if err != nil {
			return model.Job{}, fmt.Errorf("insert job file: %w", err)
		}
		job.Files = append(job.Files, saved)
	}

	if err := tx.Commit(ctx); err != nil {
		return model.Job{}, fmt.Errorf("commit transaction: %w", err)
	}

	return job, nil
}

// ListJobs returns either user-scoped history or admin-wide history from a single entrypoint.
func (r *Repository) ListJobs(ctx context.Context, actor model.User) ([]model.Job, error) {
	query := `
		SELECT id, user_id, source_type, status, output_filename, output_object_key, created_at
		FROM jobs
	`
	args := []any{}
	if actor.Role != model.RoleAdmin {
		query += ` WHERE user_id = $1`
		args = append(args, actor.ID)
	}
	query += ` ORDER BY created_at DESC`

	rows, err := r.db.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("list jobs: %w", err)
	}
	defer rows.Close()

	var jobs []model.Job
	for rows.Next() {
		var job model.Job
		if err := rows.Scan(&job.ID, &job.UserID, &job.SourceType, &job.Status, &job.OutputFilename, &job.OutputObjectKey, &job.CreatedAt); err != nil {
			return nil, fmt.Errorf("scan job: %w", err)
		}
		jobs = append(jobs, job)
	}
	return jobs, rows.Err()
}

// GetJob loads a single job with its ordered source file metadata for history detail views.
func (r *Repository) GetJob(ctx context.Context, id int64) (model.Job, error) {
	const jobQuery = `
		SELECT id, user_id, source_type, status, output_filename, output_object_key, created_at
		FROM jobs
		WHERE id = $1
	`

	var job model.Job
	err := r.db.QueryRow(ctx, jobQuery, id).
		Scan(&job.ID, &job.UserID, &job.SourceType, &job.Status, &job.OutputFilename, &job.OutputObjectKey, &job.CreatedAt)
	if err != nil {
		return model.Job{}, err
	}

	const filesQuery = `
		SELECT id, job_id, source_kind, source_name, source_order, source_size, drive_file_id, drive_link
		FROM job_files
		WHERE job_id = $1
		ORDER BY source_order ASC, source_name ASC
	`

	rows, err := r.db.Query(ctx, filesQuery, id)
	if err != nil {
		return model.Job{}, fmt.Errorf("query job files: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var file model.JobFile
		if err := rows.Scan(&file.ID, &file.JobID, &file.SourceKind, &file.SourceName, &file.SourceOrder, &file.SourceSize, &file.DriveFileID, &file.DriveLink); err != nil {
			return model.Job{}, fmt.Errorf("scan job file: %w", err)
		}
		job.Files = append(job.Files, file)
	}

	return job, rows.Err()
}

// DeleteJob removes a history record and its child file metadata together to avoid orphan rows.
func (r *Repository) DeleteJob(ctx context.Context, id int64) error {
	tx, err := r.db.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return fmt.Errorf("begin delete transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `DELETE FROM job_files WHERE job_id = $1`, id); err != nil {
		return fmt.Errorf("delete job files: %w", err)
	}
	if _, err := tx.Exec(ctx, `DELETE FROM jobs WHERE id = $1`, id); err != nil {
		return fmt.Errorf("delete job: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit delete transaction: %w", err)
	}
	return nil
}
