package main

import (
	"fmt"
	"log"
	bert_uncased "waf-detector-lib" // Refers to the local package
)

func main() {
	// Paths to assets and the shared library
	// (Adjust sharedLibPath to your local onnxruntime dylib/so location)
	modelPath := "../assets/model.onnx"
	vocabPath := "../assets/vocab.txt"
	// Ensure the user changes this path
	sharedLibPath := "../../../../venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.1.24.2.dylib"

	// 1. Initialize the Detector
	detector, err := bert_uncased.NewDetector(modelPath, vocabPath, sharedLibPath)
	if err != nil {
		log.Fatalf("Failed to initialize detector: %v", err)
	}

	// 2. Prepare request arguments (simulate a real web request)
	attackRequest := map[string]string{
		"method": "GET",
		"path":   "",
		"query":  "q=SELECT * FROM users",
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
