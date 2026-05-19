package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"waf-detector-lib/pkg/waf"
)

func main() {
	modelType := flag.String("model", "random_forest", "Model type: random_forest | logistic_regression | xgboost")
	sharedLib := flag.String("lib", "", "Path to onnxruntime shared library (e.g., libonnxruntime.dylib or .so)")
	payloadFile := flag.String("payload-file", "", "Optional path to a .txt file containing a payload to test")
	payloadContains := flag.String("payload-contains", "", "If set, select the first line containing this substring from payload-file")
	flag.Parse()

	if *sharedLib == "" {
		fmt.Println("Error: Please provide the path to the onnxruntime shared library using -lib")
		os.Exit(1)
	}

	cfg, err := waf.DefaultConfig(waf.ModelType(*modelType))
	if err != nil {
		log.Fatalf("Unknown model type: %s", *modelType)
	}

	fmt.Printf("Initializing WAF with model: %s\n", cfg.Type)

	detector, err := waf.NewModelWithConfig(cfg, *sharedLib)
	if err != nil {
		log.Fatalf("Failed to initialize model: %v", err)
	}
	defer detector.Destroy()

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
		fmt.Printf("Predict (Threshold %.2f): %v\n", cfg.PredictThreshold, isAttack)
		fmt.Printf("PredictSemantic: Blocked=%v | Score=%.4f | Reason=%s\n", blocked, semScore, reason)
		return
	}

	normalRequest := map[string]string{
		"method": "GET",
		"path":   "/api/v1/user",
		"query":  "id=123",
	}

	attackRequest := map[string]string{
		"method": "GET",
		"path":   "/search",
		"query":  "q=<script>alert('XSS')</script>",
	}

	fmt.Println("--- Simulation Start ---")

	isAttack := detector.Predict(normalRequest)
	fmt.Printf("[Stateless] Normal Request -> IsAttack: %v\n", isAttack)

	blocked, score, reason := detector.PredictSemantic("192.168.1.10", normalRequest)
	fmt.Printf("[Semantic]  Normal IP -> Blocked: %v | Score: %.2f | Reason: %s\n", blocked, score, reason)

	blocked, score, reason = detector.PredictSemantic("10.0.0.66", attackRequest)
	fmt.Printf("[Semantic]  Bad IP (Hit 1) -> Blocked: %v | Score: %.2f | Reason: %s\n", blocked, score, reason)

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
