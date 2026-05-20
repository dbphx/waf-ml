package waf

import (
	"fmt"
	"time"
)

// ModelType identifies a trained WAF classifier.
type ModelType string

const (
	ModelRandomForest        ModelType = "random_forest"
	ModelLogisticRegression  ModelType = "logistic_regression"
	ModelXGBoost             ModelType = "xgboost"
)

// Model is the public interface for ONNX-backed WAF classifiers.
type Model interface {
	Name() string
	Predict(args map[string]string) bool
	PredictScore(args map[string]string) (float64, error)
	PredictSemantic(clientIP string, args map[string]string) (blocked bool, score float64, reason string)
	SetPredictThreshold(threshold float64)
	InitReputationManager(block, suspicion float64, ttl time.Duration)
	Destroy()
}

// ModelConfig holds default thresholds and asset layout for a model.
type ModelConfig struct {
	Type                 ModelType
	AssetDir             string
	PredictThreshold     float64
	BlockThreshold       float64
	SuspicionThreshold   float64
}

// DefaultConfig returns production defaults for the given model type.
func DefaultConfig(modelType ModelType) (ModelConfig, error) {
	switch modelType {
	case ModelRandomForest:
		return ModelConfig{
			Type:               ModelRandomForest,
			AssetDir:           "random_forest/assets",
			PredictThreshold:   0.55,
			BlockThreshold:     0.55,
			SuspicionThreshold: 0.35,
		}, nil
	case ModelLogisticRegression:
		return ModelConfig{
			Type:               ModelLogisticRegression,
			AssetDir:           "logistic_regression/assets",
			PredictThreshold:   0.58,
			BlockThreshold:     0.58,
			SuspicionThreshold: 0.50,
		}, nil
	case ModelXGBoost:
		return ModelConfig{
			Type:               ModelXGBoost,
			AssetDir:           "xgboost/assets",
			PredictThreshold:   0.55,
			BlockThreshold:     0.55,
			SuspicionThreshold: 0.35,
		}, nil
	default:
		return ModelConfig{}, fmt.Errorf("unknown model type: %q", modelType)
	}
}

// NewModel creates a model implementation from type and ONNX asset directory.
func NewModel(modelType ModelType, assetDir, sharedLibPath string) (Model, error) {
	cfg, err := DefaultConfig(modelType)
	if err != nil {
		return nil, err
	}
	cfg.AssetDir = assetDir
	return NewModelWithConfig(cfg, sharedLibPath)
}

// NewModelWithConfig creates a model from an explicit configuration.
func NewModelWithConfig(cfg ModelConfig, sharedLibPath string) (Model, error) {
	switch cfg.Type {
	case ModelRandomForest:
		return newRandomForestModel(sharedLibPath, cfg)
	case ModelLogisticRegression:
		return newLogisticRegressionModel(sharedLibPath, cfg)
	case ModelXGBoost:
		return newXGBoostModel(sharedLibPath, cfg)
	default:
		return nil, fmt.Errorf("unknown model type: %q", cfg.Type)
	}
}

// baseModel wraps the shared ONNX engine and reputation manager.
type baseModel struct {
	engine     *onnxEngine
	reputation *ReputationManager
	cfg        ModelConfig
}

func (m *baseModel) Name() string {
	return string(m.cfg.Type)
}

func (m *baseModel) Predict(args map[string]string) bool {
	score, err := m.PredictScore(args)
	if err != nil {
		return false
	}
	return score >= m.engine.predictThreshold
}

func (m *baseModel) PredictScore(args map[string]string) (float64, error) {
	return m.engine.predictScore(args)
}

func (m *baseModel) SetPredictThreshold(threshold float64) {
	m.engine.setPredictThreshold(threshold)
}

func (m *baseModel) InitReputationManager(block, suspicion float64, ttl time.Duration) {
	m.reputation = NewReputationManager(m.engine, block, suspicion, ttl)
}

func (m *baseModel) PredictSemantic(clientIP string, args map[string]string) (bool, float64, string) {
	if m.reputation == nil {
		m.InitReputationManager(m.cfg.BlockThreshold, m.cfg.SuspicionThreshold, 24*time.Hour)
	}
	return m.reputation.AnalyzeRequest(clientIP, args)
}

func (m *baseModel) Destroy() {
	m.engine.destroy()
}
