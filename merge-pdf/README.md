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

## Local setup

1. Copy `.env.example` values into your shell or a local env file.
2. Start infra:

```bash
docker compose up -d postgres minio
```

3. Create database schema:

```bash
psql postgres://mergepdf:mergepdf@localhost:5432/mergepdf -f migrations/001_init.sql
psql postgres://mergepdf:mergepdf@localhost:5432/mergepdf -f scripts/seed_users.sql
```

4. Start backend:

```bash
cd backend
go mod tidy
go run ./cmd/server
```

5. Start frontend:

```bash
cd frontend
npm install
npm run dev
```

Frontend defaults to `http://localhost:5173`, backend to `http://localhost:8080`.

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
