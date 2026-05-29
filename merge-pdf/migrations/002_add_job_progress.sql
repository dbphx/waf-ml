ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS progress_percent INTEGER NOT NULL DEFAULT 0 CHECK (progress_percent >= 0 AND progress_percent <= 100);

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS error_message TEXT NULL;

ALTER TABLE jobs
    ALTER COLUMN output_object_key DROP NOT NULL;

ALTER TABLE jobs
    DROP CONSTRAINT IF EXISTS jobs_status_check;

ALTER TABLE jobs
    ADD CONSTRAINT jobs_status_check CHECK (status IN ('pending', 'running', 'completed', 'failed'));
