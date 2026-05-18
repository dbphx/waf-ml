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

// Metadata structure for the model
type Metadata struct {
	NgramRange  []int     `json:"ngram_range"`
	MaxFeatures int       `json:"max_features"`
	Vocabulary  []string  `json:"vocabulary"`
	IDF         []float64 `json:"idf"`
	Keywords    []string  `json:"keywords"`
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
		predictThreshold: 0.55, // Default to RF optimal threshold
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

	d.sessionOptions, err = ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create SessionOptions: %v", err)
	}

	d.inputName = "float_input"
	d.outputNames = []string{"label", "probabilities"}

	// Initialize buffer based on vocabulary + stats features
	featureSize := len(d.meta.Vocabulary) + 2 + len(d.meta.Keywords)
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

	texts := d.extractTextsForScoring(args)
	maxScore := 0.0
	for _, text := range texts {
		score, err := d.predictTextScoreLocked(text)
		if err != nil {
			return 0, err
		}
		if score > maxScore {
			maxScore = score
		}
	}

	return maxScore, nil
}

func (d *BaseDetector) predictTextScoreLocked(text string) (float64, error) {
	vector := d.GenerateFeatureVector(text)

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
	fields := []string{"path", "query", "headers", "body"}
	var vals []string
	for _, f := range fields {
		v := strings.TrimSpace(row[f])
		if v != "" && strings.ToLower(v) != "nan" {
			vals = append(vals, v)
		}
	}
	return strings.Join(vals, " ")
}

func (d *BaseDetector) extractTextsForScoring(row map[string]string) []string {
	combined := d.ExtractText(row)
	texts := []string{combined}
	for _, field := range []string{"path", "query", "headers", "body"} {
		value := strings.TrimSpace(row[field])
		if value != "" && strings.ToLower(value) != "nan" {
			texts = append(texts, value)
		}
	}
	return texts
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

func (d *BaseDetector) GenerateFeatureVector(text string) []float64 {
	cleaned := d.CleanText(text)

	// N-grams
	ngrams := make(map[string]int)
	chars := []rune(cleaned)
	n := len(chars)

	// Safety check for empty range
	if len(d.meta.NgramRange) < 2 {
		return make([]float64, len(d.meta.Vocabulary))
	}

	minN, maxN := d.meta.NgramRange[0], d.meta.NgramRange[1]
	for i := 0; i < n; i++ {
		for length := minN; length <= maxN; length++ {
			if i+length <= n {
				ngrams[string(chars[i:i+length])]++
			}
		}
	}

	// TF-IDF
	vector := make([]float64, len(d.meta.Vocabulary))
	for i, term := range d.meta.Vocabulary {
		if count, ok := ngrams[term]; ok {
			vector[i] = float64(count) * d.meta.IDF[i]
		}
	}

	// L2 Norm
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

	// Stats
	vector = append(vector, float64(len(text))/1000.0)
	vector = append(vector, d.CalcEntropy(text)/10.0)

	// Keywords
	for _, kw := range d.meta.Keywords {
		count := strings.Count(text, kw)
		val := 0.0
		if len(text) > 0 {
			val = float64(count) / float64(len(text)+1)
		}
		vector = append(vector, val)
	}

	return vector
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
