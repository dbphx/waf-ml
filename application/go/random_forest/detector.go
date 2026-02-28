package random_forest

import (
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"os"
	"regexp"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

type Metadata struct {
	NgramRange  []int     `json:"ngram_range"`
	MaxFeatures int       `json:"max_features"`
	Vocabulary  []string  `json:"vocabulary"`
	IDF         []float64 `json:"idf"`
	Keywords    []string  `json:"keywords"`
}

type Detector struct {
	meta           Metadata
	modelPath      string
	sessionOptions *ort.SessionOptions
	inputName      string
	outputNames    []string
}

// NewDetector initializes the WAF detector with model and metadata paths.
// sharedLibPath is the path to the onnxruntime shared library (e.g., .so or .dylib).
func NewDetector(modelPath, metaPath, sharedLibPath string) (*Detector, error) {
	d := &Detector{
		modelPath: modelPath,
	}

	// 1. Initialize ONNX runtime environment
	ort.SetSharedLibraryPath(sharedLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX: %v", err)
	}

	// 2. Load Metadata
	metaFile, err := os.Open(metaPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open metadata at %s: %v", metaPath, err)
	}
	defer metaFile.Close()

	if err := json.NewDecoder(metaFile).Decode(&d.meta); err != nil {
		return nil, fmt.Errorf("failed to decode metadata: %v", err)
	}

	// 3. Setup Session
	d.sessionOptions, err = ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create SessionOptions: %v", err)
	}

	d.inputName = "float_input"
	d.outputNames = []string{"label", "probabilities"}

	return d, nil
}

// Predict takes a map of request components and returns true if it's an ATTACK.
func (d *Detector) Predict(args map[string]string) bool {
	combined := d.ExtractText(args)
	vector := d.GenerateFeatureVector(combined)

	// Convert to float32
	vec32 := make([]float32, len(vector))
	for i, v := range vector {
		vec32[i] = float32(v)
	}

	inputShape := ort.NewShape(1, int64(len(vec32)))
	inputTensor, err := ort.NewTensor(inputShape, vec32)
	if err != nil {
		fmt.Printf("Error creating input tensor: %v\n", err)
		return false
	}
	defer inputTensor.Destroy()

	outputShape1 := ort.NewShape(1)
	outputTensor1, err := ort.NewEmptyTensor[int64](outputShape1)
	if err != nil {
		fmt.Printf("Error creating output tensor 1: %v\n", err)
		return false
	}
	defer outputTensor1.Destroy()

	outputShape2 := ort.NewShape(1, 2)
	outputTensor2, err := ort.NewEmptyTensor[float32](outputShape2)
	if err != nil {
		fmt.Printf("Error creating output tensor 2: %v\n", err)
		return false
	}
	defer outputTensor2.Destroy()

	session, err := ort.NewAdvancedSession(
		d.modelPath,
		[]string{d.inputName},
		d.outputNames,
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor1, outputTensor2},
		d.sessionOptions,
	)
	if err != nil {
		fmt.Printf("Error creating advanced session at %s: %v\n", d.modelPath, err)
		return false
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		fmt.Printf("Error running ONNX session: %v\n", err)
		return false
	}

	probsData := outputTensor2.GetData()
	attackProb := float64(probsData[1])

	// Strict 0.7 threshold as established
	return attackProb >= 0.7
}

// Internal Feature Engineering Helpers

func (d *Detector) ExtractText(row map[string]string) string {
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

func (d *Detector) CleanText(text string) string {
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

func (d *Detector) GenerateFeatureVector(text string) []float64 {
	cleaned := d.CleanText(text)

	// N-grams
	ngrams := make(map[string]int)
	chars := []rune(cleaned)
	n := len(chars)
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
		vector = append(vector, float64(count)/float64(len(text)+1))
	}

	return vector
}

func (d *Detector) CalcEntropy(text string) float64 {
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
