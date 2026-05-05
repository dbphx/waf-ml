package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"waf-detector-lib/pkg/waf"
)

func main() {
	// Parse command line arguments
	modelType := flag.String("model", "random_forest", "Model type: random_forest | logistic_regression (ONNX). securebert2 is Python-only.")
	sharedLib := flag.String("lib", "", "Path to onnxruntime shared library (e.g., libonnxruntime.dylib or .so)")
	payloadFile := flag.String("payload-file", "", "Optional path to a .txt file containing a payload to test")
	payloadContains := flag.String("payload-contains", "", "If set, select the first line containing this substring from payload-file")
	flag.Parse()

	if *modelType == "securebert2" {
		log.Fatalf("securebert2 uses Hugging Face checkpoints; use Python (src/securebert2/predict.py). Go BaseDetector only supports TF-IDF + sklearn ONNX models.")
	}

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

	// 2. Optional: Use payload from file
	if *payloadFile != "" {
		payload, err := readPayloadFromFile(*payloadFile, *payloadContains)
		if err != nil {
			log.Fatalf("Failed to read payload: %v", err)
		}

		req := map[string]string{
			"method": "GET",
			"path":   payload,
		}

		score, err := detector.PredictScore(req)
		if err != nil {
			log.Fatalf("PredictScore failed: %v", err)
		}

		isAttack := detector.Predict(req)
		blocked, semScore, reason := detector.PredictSemantic("1.1.1.1", req)

		fmt.Println("--- Single Payload Test ---")
		fmt.Printf("Payload: %s\n", payload)
		fmt.Printf("Raw Score: %.4f\n", score)
		fmt.Printf("Predict (Threshold %.2f): %v\n", blockThreshold, isAttack)
		fmt.Printf("PredictSemantic: Blocked=%v | Score=%.4f | Reason=%s\n", blocked, semScore, reason)
		return
	}

	// 3. Simulate Traffic using the two methods requested

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

func readPayloadFromFile(path string, contains string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if contains == "" || strings.Contains(line, contains) {
			return line, nil
		}
	}

	if err := scanner.Err(); err != nil {
		return "", err
	}
	if contains != "" {
		return "", fmt.Errorf("no line found containing %q", contains)
	}
	return "", fmt.Errorf("no non-empty line found in %s", path)
}
