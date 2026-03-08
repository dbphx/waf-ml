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

	// Determine paths and thresholds based on model selection
	var assetDir string
	var blockThreshold, suspicionThreshold float64

	switch *modelType {
	case "random_forest":
		assetDir = "random_forest/assets"
		// Based on analysis: Min Attack 0.64, Max Normal 0.41
		blockThreshold = 0.55
		suspicionThreshold = 0.35 // Start tracking early
	case "logistic_regression":
		assetDir = "logistic_regression/assets"
		// Based on analysis: Min Attack 0.84, Max Normal 0.59
		blockThreshold = 0.72
		suspicionThreshold = 0.50
	default:
		log.Fatalf("Unknown model type: %s. Use 'random_forest' or 'logistic_regression'", *modelType)
	}

	modelPath := fmt.Sprintf("%s/model.onnx", assetDir)
	metaPath := fmt.Sprintf("%s/model_metadata.json", assetDir)

	fmt.Printf("Initializing WAF with model: %s\n", *modelType)
	fmt.Printf("Thresholds -> Block: %.2f | Suspicion: %.2f\n", blockThreshold, suspicionThreshold)

	// 1. Initialize the Base Detector
	baseDetector, err := waf.NewBaseDetector(modelPath, metaPath, *sharedLib)
	if err != nil {
		log.Fatalf("Failed to initialize detector: %v", err)
	}

	// 2. Initialize Reputation Manager
	manager := waf.NewReputationManager(baseDetector, blockThreshold, suspicionThreshold, 24*time.Hour)

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
		"query":  "q=<script>alert('XSS')</script>", // Stronger XSS pattern
	}

	fmt.Println("--- Simulation Start ---")

	// 1. Normal Request
	blocked, score, reason := manager.AnalyzeRequest("192.168.1.10", normalRequest)
	fmt.Printf("[Normal IP] Blocked: %v | Score: %.2f | Reason: %s\n", blocked, score, reason)

	// 2. Attacker - Multiple Hits to show reputation accumulation
	ip := "10.0.0.66"
	for i := 1; i <= 5; i++ {
		blocked, score, reason = manager.AnalyzeRequest(ip, attackRequest)
		fmt.Printf("[Bad IP - Hit %d] Blocked: %v | Score: %.2f | Reason: %s\n", i, blocked, score, reason)
	}
}
