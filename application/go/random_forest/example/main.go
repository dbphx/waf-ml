package main

import (
	"fmt"
	"log"
	random_forest "waf-detector-lib" // This refers to the local package
)

func main() {
	// Paths to assets and the shared library
	// (Adjust sharedLibPath to your local onnxruntime dylib/so location)
	modelPath := "../assets/model.onnx"
	metaPath := "../assets/model_metadata.json"
	sharedLibPath := "../../../../venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.1.24.2.dylib"

	// 1. Initialize the Detector
	detector, err := random_forest.NewDetector(modelPath, metaPath, sharedLibPath)
	if err != nil {
		log.Fatalf("Failed to initialize detector: %v", err)
	}

	// 2. Prepare request arguments (simulate a real web request)
	// Interface requirement: map[string]string
	attackRequest := map[string]string{
		"method": "GET",
		"path":   "/insky/projects/bce0a79c-90d2-4558-9084-945ad6acbdae/issues",
		"query":  "db=monasca&q=SELECT%20mean(%22value%22)%20FROM%20%22mem.used_perc%22%20WHERE%20%22resource_id%22%20%3D~%20%2F%5Ebfb959fd-8128-4a68-98c7-d062ebd2dc4b%2F%20AND%20time%20%3E%3D%20now()%20-%206h%20and%20time%20%3C%3D%20now()%20GROUP%20BY%20time(1m)%2C%20resource_id&epoch=ms",
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
