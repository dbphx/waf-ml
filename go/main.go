package main

import (
	"encoding/json"
	"fmt"
	"os"
	"waf-detector/internal/features"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <http_payload_or_url>")
		os.Exit(1)
	}

	payload := os.Args[1]

	// 1. Load Metadata
	metaPath := "internal/assets/model_metadata.json"
	metaFile, err := os.Open(metaPath)
	if err != nil {
		fmt.Printf("Error: Could not open metadata at %s: %v\n", metaPath, err)
		os.Exit(1)
	}
	defer metaFile.Close()

	var meta features.Metadata
	if err := json.NewDecoder(metaFile).Decode(&meta); err != nil {
		fmt.Printf("Error: Could not decode metadata: %v\n", err)
		os.Exit(1)
	}

	// 2. Generate Features
	vector := features.GenerateFeatureVector(payload, &meta)

	// Create float32 vector since ORT often uses float32
	vec32 := make([]float32, len(vector))
	for i, v := range vector {
		vec32[i] = float32(v)
	}

	// 3. Initialize ONNX runtime and load the model
	ort.SetSharedLibraryPath("/Users/dmac/Desktop/ml/venv/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.1.24.2.dylib")
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Printf("Error initializing ONNX runtime: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	onnxPath := "internal/assets/model.onnx"

	inputShape := ort.NewShape(1, int64(len(vec32)))
	inputTensor, _ := ort.NewTensor(inputShape, vec32)
	defer inputTensor.Destroy()

	outputShape1 := ort.NewShape(1)
	outputTensor1, _ := ort.NewEmptyTensor[int64](outputShape1)
	defer outputTensor1.Destroy()

	outputShape2 := ort.NewShape(1, 2)
	outputTensor2, _ := ort.NewEmptyTensor[float32](outputShape2)
	defer outputTensor2.Destroy()

	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Printf("Error creating SessionOptions: %v\n", err)
		os.Exit(1)
	}
	defer sessionOptions.Destroy()

	session, err := ort.NewAdvancedSession(
		onnxPath,
		[]string{"float_input"},
		[]string{"label", "probabilities"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor1, outputTensor2},
		sessionOptions,
	)
	if err != nil {
		fmt.Printf("Error creating advanced session: %v\n", err)
		os.Exit(1)
	}
	defer session.Destroy()

	err = session.Run()
	if err != nil {
		fmt.Printf("Error running ONNX session: %v\n", err)
		os.Exit(1)
	}

	labels := outputTensor1.GetData()
	probsData := outputTensor2.GetData()

	prediction := "NORMAL"
	if labels[0] == 1 {
		prediction = "ATTACK"
	}

	prob := float64(probsData[1])
	confidence := prob
	if prob < 0.7 {
		confidence = 1.0 - prob
	}

	fmt.Printf("Payload: %s\n", payload)
	fmt.Printf("Prediction: %s\n", prediction)
	fmt.Printf("Confidence: %.4f\n", confidence)
}
