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
		// Try to find a default or error out
		fmt.Println("Error: Please provide the path to the onnxruntime shared library using -lib")
		fmt.Println("Example: go run example.go -model random_forest -lib /path/to/libonnxruntime.1.14.1.dylib")
		os.Exit(1)
	}

	// Determine paths based on model selection
	var assetDir string
	switch *modelType {
	case "random_forest":
		assetDir = "random_forest/assets"
	case "logistic_regression":
		assetDir = "logistic_regression/assets"
	default:
		log.Fatalf("Unknown model type: %s. Use 'random_forest' or 'logistic_regression'", *modelType)
	}

	modelPath := fmt.Sprintf("%s/model.onnx", assetDir)
	metaPath := fmt.Sprintf("%s/model_metadata.json", assetDir)

	fmt.Printf("Initializing WAF with model: %s\n", *modelType)
	fmt.Printf("Model Path: %s\n", modelPath)

	// 1. Initialize the Base Detector
	baseDetector, err := waf.NewBaseDetector(modelPath, metaPath, *sharedLib)
	if err != nil {
		log.Fatalf("Failed to initialize detector: %v", err)
	}

	// 2. Initialize Reputation Manager
	// Block Threshold: 0.8
	// Suspicion Threshold: 0.5
	// TTL: 24 Hours
	manager := waf.NewReputationManager(baseDetector, 0.8, 0.5, 24*time.Hour)

	// 3. Simulate Traffic

	// Case A: Normal User (Clean)
	normalRequest := map[string]string{
		"method": "GET",
		"path":   "/api/v1/user",
		"query":  "id=123",
	}

	// Case B: Attacker (Initially suspicious, then blocked)
	attackRequest := map[string]string{
		"method": "GET",
		"path":   "/search",
		"query":  "q=union+select+user", // Classic SQLi pattern
	}

	fmt.Println("--- Simulation Start ---")

	// 1. Normal Request
	blocked, score, reason := manager.AnalyzeRequest("192.168.1.10", normalRequest)
	fmt.Printf("[Normal IP] Blocked: %v | Score: %.2f | Reason: %s\n", blocked, score, reason)

	// 2. Attacker - First Hit (Maybe just suspicious)
	blocked, score, reason = manager.AnalyzeRequest("10.0.0.66", attackRequest)
	fmt.Printf("[Bad IP - Hit 1] Blocked: %v | Score: %.2f | Reason: %s\n", blocked, score, reason)

	// 3. Attacker - Second Hit (Should be blocked due to reputation accumulation)
	// We use the same request, but the memory should kick in
	blocked, score, reason = manager.AnalyzeRequest("10.0.0.66", attackRequest)
	fmt.Printf("[Bad IP - Hit 2] Blocked: %v | Score: %.2f | Reason: %s\n", blocked, score, reason)
}
