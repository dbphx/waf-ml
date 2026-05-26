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
	ModelName         string                     `json:"model_name"`
	FieldOrder        []string                   `json:"field_order"`
	FieldVectorizers  map[string]FieldVectorizer `json:"field_vectorizers"`
	Keywords          []string                   `json:"keywords"`
	KeywordMatchModes map[string]string          `json:"keyword_match_modes"`
}

// featureEncoder maps a feature value into the ONNX input tensor.
type featureEncoder interface {
	encode(index int, value float64) float32
}

type denseEncoder struct{}

func (denseEncoder) encode(_ int, value float64) float32 {
	return float32(value)
}

var _ Scorer = (*onnxEngine)(nil)

// xgboostEncoder treats zero TF-IDF features as missing (NaN), matching sparse training.
type xgboostEncoder struct{}

func (xgboostEncoder) encode(_ int, value float64) float32 {
	if value == 0 {
		return float32(math.NaN())
	}
	return float32(value)
}

// onnxEngine holds the ONNX session and shared feature engineering.
type onnxEngine struct {
	meta             Metadata
	modelPath        string
	encoder          featureEncoder
	sessionOptions   *ort.SessionOptions
	session          *ort.AdvancedSession
	inputName        string
	outputNames      []string
	inputTensor      *ort.Tensor[float32]
	outputTensor1    *ort.Tensor[int64]
	outputTensor2    *ort.Tensor[float32]
	inputBuffer      []float32
	mu               sync.Mutex
	predictThreshold float64
}

func newOnnxEngine(modelPath, metaPath, sharedLibPath string, encoder featureEncoder, predictThreshold float64) (*onnxEngine, error) {
	e := &onnxEngine{
		modelPath:        modelPath,
		encoder:          encoder,
		predictThreshold: predictThreshold,
	}

	ort.SetSharedLibraryPath(sharedLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("initialize ONNX: %w", err)
	}

	metaFile, err := os.Open(metaPath)
	if err != nil {
		return nil, fmt.Errorf("open metadata: %w", err)
	}
	defer metaFile.Close()

	if err := json.NewDecoder(metaFile).Decode(&e.meta); err != nil {
		return nil, fmt.Errorf("decode metadata: %w", err)
	}
	if len(e.meta.FieldOrder) == 0 {
		return nil, fmt.Errorf("invalid model metadata: missing field_order")
	}
	for _, field := range e.meta.FieldOrder {
		if _, ok := e.meta.FieldVectorizers[field]; !ok {
			return nil, fmt.Errorf("invalid model metadata: missing field_vectorizer for %q", field)
		}
	}

	e.sessionOptions, err = ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}

	e.inputName = "float_input"
	e.outputNames = []string{"label", "probabilities"}

	featureSize := e.featureVectorSize()
	if featureSize == 0 {
		return nil, fmt.Errorf("invalid model metadata: empty multipart feature layout")
	}
	e.inputBuffer = make([]float32, featureSize)

	inputShape := ort.NewShape(1, int64(featureSize))
	e.inputTensor, err = ort.NewTensor(inputShape, e.inputBuffer)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	outputShape1 := ort.NewShape(1)
	e.outputTensor1, err = ort.NewEmptyTensor[int64](outputShape1)
	if err != nil {
		return nil, fmt.Errorf("create output tensor 1: %w", err)
	}

	outputShape2 := ort.NewShape(1, 2)
	e.outputTensor2, err = ort.NewEmptyTensor[float32](outputShape2)
	if err != nil {
		return nil, fmt.Errorf("create output tensor 2: %w", err)
	}

	e.session, err = ort.NewAdvancedSession(
		e.modelPath,
		[]string{e.inputName},
		e.outputNames,
		[]ort.ArbitraryTensor{e.inputTensor},
		[]ort.ArbitraryTensor{e.outputTensor1, e.outputTensor2},
		e.sessionOptions,
	)
	if err != nil {
		return nil, fmt.Errorf("create ONNX session: %w", err)
	}

	return e, nil
}

func (e *onnxEngine) destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
	if e.inputTensor != nil {
		e.inputTensor.Destroy()
	}
	if e.outputTensor1 != nil {
		e.outputTensor1.Destroy()
	}
	if e.outputTensor2 != nil {
		e.outputTensor2.Destroy()
	}
	if e.sessionOptions != nil {
		e.sessionOptions.Destroy()
	}
}

func (e *onnxEngine) setPredictThreshold(threshold float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.predictThreshold = threshold
}

// PredictScore implements Scorer.
func (e *onnxEngine) PredictScore(args map[string]string) (float64, error) {
	return e.predictScore(args)
}

// Predict implements Scorer.
func (e *onnxEngine) Predict(args map[string]string) bool {
	return e.predict(args)
}

func (e *onnxEngine) predictScore(args map[string]string) (float64, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	rows := e.extractRowsForScoring(args)
	maxScore := 0.0
	for _, row := range rows {
		score, err := e.predictRowScoreLocked(row)
		if err != nil {
			return 0, err
		}
		if score > maxScore {
			maxScore = score
		}
	}
	return maxScore, nil
}

func (e *onnxEngine) predict(args map[string]string) bool {
	score, err := e.predictScore(args)
	if err != nil {
		return false
	}
	return score >= e.predictThreshold
}

func (e *onnxEngine) predictRowScoreLocked(row map[string]string) (float64, error) {
	vector := e.generateFeatureVector(row)
	if len(vector) != len(e.inputBuffer) {
		return 0, fmt.Errorf("feature vector size mismatch: expected %d, got %d", len(e.inputBuffer), len(vector))
	}
	for i, v := range vector {
		e.inputBuffer[i] = e.encoder.encode(i, v)
	}
	if err := e.session.Run(); err != nil {
		return 0, err
	}
	probsData := e.outputTensor2.GetData()
	return float64(probsData[1]), nil
}

func (e *onnxEngine) extractRowsForScoring(row map[string]string) []map[string]string {
	fields := e.meta.FieldOrder
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

func (e *onnxEngine) cleanText(text string) string {
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

func (e *onnxEngine) generateFeatureVector(row map[string]string) []float64 {
	fields := e.meta.FieldOrder
	if len(fields) == 0 {
		fields = []string{"path", "query", "headers", "body"}
	}

	vector := make([]float64, 0, e.featureVectorSize())
	for _, field := range fields {
		vectorizer := e.meta.FieldVectorizers[field]
		vector = append(vector, e.generateFieldTFIDF(vectorizer, row[field])...)
	}
	for _, field := range fields {
		vector = append(vector, e.generateFieldStats(row[field])...)
	}
	return vector
}

func (e *onnxEngine) featureVectorSize() int {
	total := 0
	for _, field := range e.meta.FieldOrder {
		total += len(e.meta.FieldVectorizers[field].Vocabulary)
	}
	total += len(e.meta.FieldOrder) * (2 + len(e.meta.Keywords))
	return total
}

func (e *onnxEngine) generateFieldTFIDF(vectorizer FieldVectorizer, text string) []float64 {
	vector := make([]float64, len(vectorizer.Vocabulary))
	if len(vectorizer.NgramRange) < 2 || len(vectorizer.Vocabulary) == 0 {
		return vector
	}

	cleaned := e.cleanText(text)
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

func (e *onnxEngine) generateFieldStats(text string) []float64 {
	stats := make([]float64, 0, 2+len(e.meta.Keywords))
	stats = append(stats, float64(len(text))/1000.0)
	stats = append(stats, e.calcEntropy(text)/10.0)
	for _, kw := range e.meta.Keywords {
		count := e.countKeywordOccurrences(text, kw)
		val := 0.0
		if len(text) > 0 {
			val = float64(count) / float64(len(text)+1)
		}
		stats = append(stats, val)
	}
	return stats
}

func (e *onnxEngine) countKeywordOccurrences(text, keyword string) int {
	if e.meta.KeywordMatchModes[keyword] == "token" {
		pattern := regexp.MustCompile(fmt.Sprintf(`(?i)(?<![a-z])%s(?![a-z])`, regexp.QuoteMeta(keyword)))
		return len(pattern.FindAllStringIndex(text, -1))
	}
	return strings.Count(text, keyword)
}

func (e *onnxEngine) calcEntropy(text string) float64 {
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

func newBaseModel(cfg ModelConfig, sharedLibPath string, encoder featureEncoder) (*baseModel, error) {
	modelPath := fmt.Sprintf("%s/model.onnx", cfg.AssetDir)
	metaPath := fmt.Sprintf("%s/model_metadata.json", cfg.AssetDir)

	engine, err := newOnnxEngine(modelPath, metaPath, sharedLibPath, encoder, cfg.PredictThreshold)
	if err != nil {
		return nil, err
	}

	m := &baseModel{
		engine: engine,
		cfg:    cfg,
	}
	m.InitReputationManager(cfg.BlockThreshold, cfg.SuspicionThreshold, 24*time.Hour)
	return m, nil
}
