package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type Config struct {
	Port              string
	PublicURL         string
	DatabaseURL       string
	JWTSecret         string
	MaxUploadBytes    int64
	RequestTimeout    time.Duration
	GoogleDriveAPIKey string
	MinIOEndpoint     string
	MinIOAccessKey    string
	MinIOSecretKey    string
	MinIOBucket       string
	MinIOUseSSL       bool
	AllowedOrigin     string
}

// Load centralizes runtime configuration so local dev and production boot the same way.
func Load() (Config, error) {
	port := envOrDefault("BACKEND_PORT", "8080")
	publicURL := envOrDefault("BACKEND_PUBLIC_URL", "http://localhost:"+port)
	maxUploadMB, err := strconv.Atoi(envOrDefault("MAX_UPLOAD_MB", "25"))
	if err != nil {
		return Config{}, fmt.Errorf("parse MAX_UPLOAD_MB: %w", err)
	}

	timeoutSeconds, err := strconv.Atoi(envOrDefault("REQUEST_TIMEOUT_SECONDS", "60"))
	if err != nil {
		return Config{}, fmt.Errorf("parse REQUEST_TIMEOUT_SECONDS: %w", err)
	}

	minioSSL, err := strconv.ParseBool(envOrDefault("MINIO_USE_SSL", "false"))
	if err != nil {
		return Config{}, fmt.Errorf("parse MINIO_USE_SSL: %w", err)
	}

	cfg := Config{
		Port:              port,
		PublicURL:         publicURL,
		DatabaseURL:       envOrDefault("DATABASE_URL", "postgres://mergepdf:mergepdf@localhost:5432/mergepdf?sslmode=disable"),
		JWTSecret:         os.Getenv("JWT_SECRET"),
		MaxUploadBytes:    int64(maxUploadMB) * 1024 * 1024,
		RequestTimeout:    time.Duration(timeoutSeconds) * time.Second,
		GoogleDriveAPIKey: os.Getenv("GOOGLE_DRIVE_API_KEY"),
		MinIOEndpoint:     envOrDefault("MINIO_ENDPOINT", "localhost:9000"),
		MinIOAccessKey:    envOrDefault("MINIO_ACCESS_KEY", "minioadmin"),
		MinIOSecretKey:    envOrDefault("MINIO_SECRET_KEY", "minioadmin"),
		MinIOBucket:       envOrDefault("MINIO_BUCKET", "merged-pdfs"),
		MinIOUseSSL:       minioSSL,
		AllowedOrigin:     envOrDefault("ALLOWED_ORIGIN", "http://localhost:5173"),
	}

	if cfg.JWTSecret == "" {
		return Config{}, fmt.Errorf("JWT_SECRET is required")
	}

	return cfg, nil
}

func envOrDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
