package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"waf-detector-lib/pkg/waf"
)

func main() {
	// Parse command line arguments
	modelType := flag.String("model", "random_forest", "Model type to use: 'random_forest' or 'logistic_regression'")
	sharedLib := flag.String("lib", "", "Path to onnxruntime shared library (e.g., libonnxruntime.dylib or .so)")
	flag.Parse()

	if *sharedLib == "" {
		fmt.Println("Error: Please provide the path to the onnxruntime shared library using -lib")
		os.Exit(1)
	}

	// Determine paths
	var assetDir string
	var blockThreshold, suspicionThreshold float64

	switch *modelType {
	case "random_forest":
		assetDir = "random_forest/assets"
		blockThreshold = 0.55
		suspicionThreshold = 0.35
	case "logistic_regression":
		assetDir = "logistic_regression/assets"
		blockThreshold = 0.72
		suspicionThreshold = 0.50
	default:
		log.Fatalf("Unknown model type: %s", *modelType)
	}

	modelPath := fmt.Sprintf("%s/model.onnx", assetDir)
	metaPath := fmt.Sprintf("%s/model_metadata.json", assetDir)

	fmt.Printf("Initializing WAF with model: %s\n", *modelType)

	// 1. Initialize the Base Detector
	// This now internally manages the single ONNX session and default Reputation Manager
	detector, err := waf.NewBaseDetector(modelPath, metaPath, *sharedLib)
	if err != nil {
		log.Fatalf("Failed to initialize detector: %v", err)
	}
	defer detector.Destroy()

	// Apply correct thresholds based on the model chosen
	detector.SetPredictThreshold(blockThreshold)
	// We re-initialize the internal reputation manager to use the correct model-specific thresholds
	detector.InitReputationManager(blockThreshold, suspicionThreshold, 24*time.Hour)

	// 2. Simulate Traffic using the two methods requested

	// Case A: Normal User
	normalRequest := map[string]string{
		"method": "GET",
		"path":   "/api/v1/user",
		"query":  "id=123",
	}

	// Case B: Attacker
	attackRequest := map[string]string{
		"method": "GET",
		"path":   "/search",
		"query":  "q=<script>alert('XSS')</script>",
	}

	fmt.Println("--- Simulation Start ---")

	// METHOD 1: Predict (Stateless, Boolean)
	isAttack := detector.Predict(normalRequest)
	fmt.Printf("[Stateless] Normal Request -> IsAttack: %v\n", isAttack)

	// METHOD 2: PredictSemantic (Stateful, Score + Reputation)
	// Normal IP
	blocked, score, reason := detector.PredictSemantic("192.168.1.10", normalRequest)
	fmt.Printf("[Semantic]  Normal IP -> Blocked: %v | Score: %.2f | Reason: %s\n", blocked, score, reason)

	// Attacker IP - Hit 1
	blocked, score, reason = detector.PredictSemantic("10.0.0.66", attackRequest)
	fmt.Printf("[Semantic]  Bad IP (Hit 1) -> Blocked: %v | Score: %.2f | Reason: %s\n", blocked, score, reason)

	// Attacker IP - Hit 2 (Reputation kicking in)
	blocked, score, reason = detector.PredictSemantic("10.0.0.66", attackRequest)
	fmt.Printf("[Semantic]  Bad IP (Hit 2) -> Blocked: %v | Score: %.2f | Reason: %s\n", blocked, score, reason)
}
