package main

import (
	"fmt"
	"log"
	logistic_regression "waf-detector-lib" // This refers to the local package
)

func main() {
	// Paths to assets and the shared library
	// (Adjust sharedLibPath to your local onnxruntime dylib/so location)
	modelPath := "../assets/model.onnx"
	metaPath := "../assets/model_metadata.json"
	sharedLibPath := "../../../../venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.1.24.2.dylib"

	// 1. Initialize the Detector
	detector, err := logistic_regression.NewDetector(modelPath, metaPath, sharedLibPath)
	if err != nil {
		log.Fatalf("Failed to initialize detector: %v", err)
	}

	// 2. Prepare request arguments (simulate a real web request)
	// Interface requirement: map[string]string
	attackRequest := map[string]string{
		"method": "GET",
		"path":   "/search",
		"query":  "q=apple' OR '1'='1",
		"body":   "",
	}

	normalRequest := map[string]string{
		"method": "GET",
		"path":   "/api/v1/user",
		"query":  "id=123",
		"body":   "",
	}

	// 3. Predict
	isAttack1 := detector.Predict(attackRequest)
	isAttack2 := detector.Predict(normalRequest)

	fmt.Printf("Request 1 (SQLi): Attack Detected? %v\n", isAttack1)
	fmt.Printf("Request 2 (Safe): Attack Detected? %v\n", isAttack2)
}
