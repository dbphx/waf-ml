package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ml/merge-pdf/backend/internal/auth"
	"github.com/ml/merge-pdf/backend/internal/config"
	"github.com/ml/merge-pdf/backend/internal/drive"
	"github.com/ml/merge-pdf/backend/internal/repository"
	"github.com/ml/merge-pdf/backend/internal/server"
	"github.com/ml/merge-pdf/backend/internal/storage"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	ctx := context.Background()
	db, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("connect database: %v", err)
	}
	defer db.Close()

	minioClient, err := storage.New(cfg.MinIOEndpoint, cfg.MinIOAccessKey, cfg.MinIOSecretKey, cfg.MinIOBucket, cfg.MinIOUseSSL)
	if err != nil {
		log.Fatalf("create storage client: %v", err)
	}
	if err := minioClient.EnsureBucket(ctx); err != nil {
		log.Fatalf("ensure storage bucket: %v", err)
	}

	repo := repository.New(db)
	authSvc := auth.NewService(cfg.JWTSecret)
	driveClient := drive.NewClient(cfg.GoogleDriveAPIKey, http.DefaultClient)
	srv := server.New(cfg, repo, authSvc, driveClient, minioClient)

	shutdownCtx, stop := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		<-shutdownCtx.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		srv.Shutdown(ctx)
	}()

	if err := srv.Start(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("server error: %v", err)
	}
}
