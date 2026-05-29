# Merge PDF Workspace

Authenticated PDF merge app with:

- `React` frontend
- `Go` backend API
- `PostgreSQL` for users and job history
- `MinIO` for merged output storage
- Google Drive shared-folder ingest ordered by numeric prefixes in filenames

## Features

- Login with seeded email/password accounts
- Merge PDFs from a shared Google Drive folder
- Merge uploaded local PDFs with editable order
- Store merged outputs in MinIO
- View, download, and delete merge history per user
- Show progress percent while a merge job is still running

## Local setup

1. Copy `.env.example` values into your shell or a local env file.
2. Start the full stack:

```bash
docker compose up -d --build
```

Postgres auto-runs the schema and seed scripts on first boot.
If your database volume already exists from an older version, apply the progress migration manually:

```bash
psql postgres://mergepdf:mergepdf@localhost:5432/mergepdf -f migrations/002_add_job_progress.sql
```

Services:

- Frontend: `http://localhost:4173`
- Backend API: `http://localhost:8080`
- MinIO console: `http://localhost:9001`
- Postgres: `localhost:5432`

If you need to rebuild from a clean database state, remove the persisted volumes first:

```bash
docker compose down -v
docker compose up -d --build
```

## Required environment

- `JWT_SECRET`
- `DATABASE_URL`
- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET`
- `GOOGLE_DRIVE_API_KEY`

`GOOGLE_DRIVE_API_KEY` is required for shared/public Drive folder preview and download.

## Seed users

`scripts/seed_users.sql` creates:

- `admin@example.com`
- `user@example.com`

Default password for both: `ChangeMe123!`

## API summary

- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/me`
- `POST /api/drive/preview`
- `POST /api/merge/drive`
- `POST /api/merge/upload`
- `GET /api/jobs`
- `GET /api/jobs/:id`
- `GET /api/jobs/:id/download`
- `DELETE /api/jobs/:id`

## Drive ordering

Drive merges only use the first integer found in each filename:

- `1-cover.pdf`
- `02-chapter.pdf`
- `10-appendix.pdf`

Files without a number in the name are rejected.
