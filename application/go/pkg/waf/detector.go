package waf

import (
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// FieldVectorizer stores the per-request-part TF-IDF parameters exported by Python.
type FieldVectorizer struct {
	NgramRange  []int     `json:"ngram_range"`
	MaxFeatures int       `json:"max_features"`
	Vocabulary  []string  `json:"vocabulary"`
	IDF         []float64 `json:"idf"`
}

// Metadata describes the multipart feature layout consumed by the ONNX model.
type Metadata struct {
	ModelName        string                     `json:"model_name"`
	FieldOrder       []string                   `json:"field_order"`
	FieldVectorizers map[string]FieldVectorizer `json:"field_vectorizers"`
	Keywords         []string                   `json:"keywords"`
}

// BaseDetector holds the session and resources for ONNX inference
type BaseDetector struct {
	meta           Metadata
	modelPath      string
	sessionOptions *ort.SessionOptions
	session        *ort.AdvancedSession
	inputName      string
	outputNames    []string

	// Reuse tensors to avoid allocation per request
	inputTensor   *ort.Tensor[float32]
	outputTensor1 *ort.Tensor[int64]
	outputTensor2 *ort.Tensor[float32]
	inputBuffer   []float32

	// Mutex to ensure thread safety when using the single session/buffer
	mu sync.Mutex

	// Reputation Manager (Optional, initialized if semantic check is needed)
	reputation *ReputationManager

	// Default threshold for stateless Predict()
	predictThreshold float64
}

// NewBaseDetector initializes a detector and creates the ONNX session once.
// It also initializes the shared ReputationManager with default settings.
func NewBaseDetector(modelPath, metaPath, sharedLibPath string) (*BaseDetector, error) {
	d := &BaseDetector{
		modelPath:        modelPath,
		predictThreshold: 0.55,
	}

	ort.SetSharedLibraryPath(sharedLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX: %v", err)
	}

	metaFile, err := os.Open(metaPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open metadata: %v", err)
	}
	defer metaFile.Close()

	if err := json.NewDecoder(metaFile).Decode(&d.meta); err != nil {
		return nil, fmt.Errorf("failed to decode metadata: %v", err)
	}
	switch d.meta.ModelName {
	case "logistic_regression":
		d.predictThreshold = 0.77
	case "random_forest":
		d.predictThreshold = 0.55
	}
	if len(d.meta.FieldOrder) == 0 {
		return nil, fmt.Errorf("invalid model metadata: missing field_order")
	}
	for _, field := range d.meta.FieldOrder {
		if _, ok := d.meta.FieldVectorizers[field]; !ok {
			return nil, fmt.Errorf("invalid model metadata: missing field_vectorizer for %q", field)
		}
	}

	d.sessionOptions, err = ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create SessionOptions: %v", err)
	}

	d.inputName = "float_input"
	d.outputNames = []string{"label", "probabilities"}

	featureSize := d.featureVectorSize()
	if featureSize == 0 {
		return nil, fmt.Errorf("invalid model metadata: empty multipart feature layout")
	}
	d.inputBuffer = make([]float32, featureSize)

	// Create Tensors ONCE
	inputShape := ort.NewShape(1, int64(featureSize))
	d.inputTensor, err = ort.NewTensor(inputShape, d.inputBuffer)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}

	outputShape1 := ort.NewShape(1)
	d.outputTensor1, err = ort.NewEmptyTensor[int64](outputShape1)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor 1: %v", err)
	}

	outputShape2 := ort.NewShape(1, 2)
	d.outputTensor2, err = ort.NewEmptyTensor[float32](outputShape2)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor 2: %v", err)
	}

	// Create Session ONCE bound to these tensors
	d.session, err = ort.NewAdvancedSession(
		d.modelPath,
		[]string{d.inputName},
		d.outputNames,
		[]ort.ArbitraryTensor{d.inputTensor},
		[]ort.ArbitraryTensor{d.outputTensor1, d.outputTensor2},
		d.sessionOptions,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %v", err)
	}

	// Initialize default Reputation Manager (can be configured later if needed)
	// Default: Block 0.8, Suspicion 0.5, TTL 24h
	d.reputation = NewReputationManager(d, 0.8, 0.5, 24*time.Hour)

	return d, nil
}

// Destroy cleans up ONNX resources
func (d *BaseDetector) Destroy() {
	if d.session != nil {
		d.session.Destroy()
	}
	if d.inputTensor != nil {
		d.inputTensor.Destroy()
	}
	if d.outputTensor1 != nil {
		d.outputTensor1.Destroy()
	}
	if d.outputTensor2 != nil {
		d.outputTensor2.Destroy()
	}
	if d.sessionOptions != nil {
		d.sessionOptions.Destroy()
	}
}

// PredictScore calculates the raw probability of an attack (Stateless)
// Thread-safe due to internal mutex
func (d *BaseDetector) PredictScore(args map[string]string) (float64, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	rows := d.extractRowsForScoring(args)
	maxScore := 0.0
	for _, row := range rows {
		score, err := d.predictRowScoreLocked(row)
		if err != nil {
			return 0, err
		}
		if score > maxScore {
			maxScore = score
		}
	}

	return maxScore, nil
}

func (d *BaseDetector) predictRowScoreLocked(row map[string]string) (float64, error) {
	vector := d.GenerateFeatureVector(row)

	// Update the input buffer in-place
	// Since d.inputBuffer is the backing slice for d.inputTensor, updating this updates the tensor data
	if len(vector) != len(d.inputBuffer) {
		return 0, fmt.Errorf("feature vector size mismatch: expected %d, got %d", len(d.inputBuffer), len(vector))
	}
	for i, v := range vector {
		d.inputBuffer[i] = float32(v)
	}

	// Run inference
	if err := d.session.Run(); err != nil {
		return 0, err
	}

	probsData := d.outputTensor2.GetData()
	// Index 1 contains the probability of class 1 (ATTACK)
	return float64(probsData[1]), nil
}

// SetPredictThreshold updates the threshold used by the stateless Predict method.
func (d *BaseDetector) SetPredictThreshold(threshold float64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.predictThreshold = threshold
}

// Predict implements the legacy boolean interface using the configured threshold.
func (d *BaseDetector) Predict(args map[string]string) bool {
	score, err := d.PredictScore(args)
	if err != nil {
		return false
	}
	return score >= d.predictThreshold
}

// InitReputationManager allows re-configuring the built-in reputation manager
func (d *BaseDetector) InitReputationManager(block, suspicion float64, ttl time.Duration) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.reputation = NewReputationManager(d, block, suspicion, ttl)
}

// PredictSemantic implements the stateful check using the Reputation System.
// It returns (isBlocked, score, reason).
func (d *BaseDetector) PredictSemantic(clientIP string, args map[string]string) (bool, float64, string) {
	// Use the internal ReputationManager which calls back to d.PredictScore
	return d.reputation.AnalyzeRequest(clientIP, args)
}

// --- Helpers ---

func (d *BaseDetector) ExtractText(row map[string]string) string {
	fields := d.meta.FieldOrder
	if len(fields) == 0 {
		fields = []string{"path", "query", "headers", "body"}
	}
	var vals []string
	for _, f := range fields {
		v := strings.TrimSpace(row[f])
		if v != "" && strings.ToLower(v) != "nan" {
			vals = append(vals, v)
		}
	}
	return strings.Join(vals, " ")
}

func (d *BaseDetector) extractRowsForScoring(row map[string]string) []map[string]string {
	fields := d.meta.FieldOrder
	if len(fields) == 0 {
		fields = []string{"path", "query", "headers", "body"}
	}

	rows := make([]map[string]string, 0, len(fields))
	for _, field := range fields {
		value := strings.TrimSpace(row[field])
		if value != "" && strings.ToLower(value) != "nan" {
			fieldRow := make(map[string]string, len(fields))
			for _, fieldName := range fields {
				fieldRow[fieldName] = ""
			}
			fieldRow[field] = value
			rows = append(rows, fieldRow)
		}
	}

	if len(rows) == 0 {
		fallback := make(map[string]string, len(fields))
		for _, field := range fields {
			fallback[field] = ""
		}
		fallback["path"] = "/"
		rows = append(rows, fallback)
	}

	return rows
}

func (d *BaseDetector) CleanText(text string) string {
	text = strings.ToLower(text)
	decoded, err := url.PathUnescape(text)
	if err == nil {
		text = decoded
		decoded, err = url.PathUnescape(text)
		if err == nil {
			text = decoded
		}
	}
	re := regexp.MustCompile(`\s+`)
	text = re.ReplaceAllString(text, " ")
	return strings.TrimSpace(text)
}

func (d *BaseDetector) GenerateFeatureVector(row map[string]string) []float64 {
	fields := d.meta.FieldOrder
	if len(fields) == 0 {
		fields = []string{"path", "query", "headers", "body"}
	}

	vector := make([]float64, 0, d.featureVectorSize())

	for _, field := range fields {
		vectorizer := d.meta.FieldVectorizers[field]
		vector = append(vector, d.generateFieldTFIDF(vectorizer, row[field])...)
	}

	for _, field := range fields {
		vector = append(vector, d.generateFieldStats(row[field])...)
	}

	return vector
}

func (d *BaseDetector) featureVectorSize() int {
	total := 0
	for _, field := range d.meta.FieldOrder {
		total += len(d.meta.FieldVectorizers[field].Vocabulary)
	}
	total += len(d.meta.FieldOrder) * (2 + len(d.meta.Keywords))
	return total
}

func (d *BaseDetector) generateFieldTFIDF(vectorizer FieldVectorizer, text string) []float64 {
	vector := make([]float64, len(vectorizer.Vocabulary))
	if len(vectorizer.NgramRange) < 2 || len(vectorizer.Vocabulary) == 0 {
		return vector
	}

	cleaned := d.CleanText(text)
	ngrams := make(map[string]int)
	chars := []rune(cleaned)
	minN, maxN := vectorizer.NgramRange[0], vectorizer.NgramRange[1]
	for i := 0; i < len(chars); i++ {
		for length := minN; length <= maxN; length++ {
			if i+length <= len(chars) {
				ngrams[string(chars[i:i+length])]++
			}
		}
	}

	for i, term := range vectorizer.Vocabulary {
		if i >= len(vectorizer.IDF) {
			break
		}
		if count, ok := ngrams[term]; ok {
			vector[i] = float64(count) * vectorizer.IDF[i]
		}
	}

	var sumSq float64
	for _, v := range vector {
		sumSq += v * v
	}
	if sumSq > 0 {
		norm := math.Sqrt(sumSq)
		for i := range vector {
			vector[i] /= norm
		}
	}

	return vector
}

func (d *BaseDetector) generateFieldStats(text string) []float64 {
	stats := make([]float64, 0, 2+len(d.meta.Keywords))
	stats = append(stats, float64(len(text))/1000.0)
	stats = append(stats, d.CalcEntropy(text)/10.0)
	for _, kw := range d.meta.Keywords {
		count := strings.Count(text, kw)
		val := 0.0
		if len(text) > 0 {
			val = float64(count) / float64(len(text)+1)
		}
		stats = append(stats, val)
	}
	return stats
}

func (d *BaseDetector) CalcEntropy(text string) float64 {
	if len(text) == 0 {
		return 0
	}
	counts := make(map[rune]int)
	for _, r := range text {
		counts[r]++
	}
	var entropy float64
	total := float64(len(text))
	for _, count := range counts {
		p := float64(count) / total
		entropy -= p * math.Log(p)
	}
	return entropy
}
