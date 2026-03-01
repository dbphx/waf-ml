package bert_uncased

import (
	"bufio"
	"fmt"
	"math"
	"net/url"
	"os"
	"regexp"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

type Detector struct {
	modelPath      string
	sessionOptions *ort.SessionOptions
	vocab          map[string]int
	maxLen         int
}

// NewDetector initializes the WAF detector with model and vocab paths.
// sharedLibPath is the path to the onnxruntime shared library.
func NewDetector(modelPath, vocabPath, sharedLibPath string) (*Detector, error) {
	d := &Detector{
		modelPath: modelPath,
		maxLen:    128,
		vocab:     make(map[string]int),
	}

	// 1. Initialize ONNX runtime environment
	ort.SetSharedLibraryPath(sharedLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX: %v", err)
	}

	// 2. Load Vocabulary (vocab.txt)
	file, err := os.Open(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open vocab %s: %v", vocabPath, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	idx := 0
	for scanner.Scan() {
		word := scanner.Text()
		d.vocab[word] = idx
		idx++
	}

	// 3. Setup Session
	d.sessionOptions, err = ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create SessionOptions: %v", err)
	}

	return d, nil
}

// Predict takes a map of request components and returns true if it's an ATTACK.
func (d *Detector) Predict(args map[string]string) bool {
	combined := d.ExtractText(args)
	inputIDs, attentionMask := d.Tokenize(combined)

	inputShape := ort.NewShape(1, int64(d.maxLen))

	// Create Tensors
	inputTensor, err := ort.NewTensor(inputShape, inputIDs)
	if err != nil {
		fmt.Printf("Error creating input_ids tensor: %v\n", err)
		return false
	}
	defer inputTensor.Destroy()

	maskTensor, err := ort.NewTensor(inputShape, attentionMask)
	if err != nil {
		fmt.Printf("Error creating attention_mask tensor: %v\n", err)
		return false
	}
	defer maskTensor.Destroy()

	outputShape := ort.NewShape(1, 2)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		fmt.Printf("Error creating output tensor: %v\n", err)
		return false
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(
		d.modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		[]ort.ArbitraryTensor{inputTensor, maskTensor},
		[]ort.ArbitraryTensor{outputTensor},
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

	logits := outputTensor.GetData()

	// Apply Softmax
	// The Output shape is [1, 2], so logits contains exactly two float32 values.
	// In the PyTorch model, index 0 -> NORMAL, index 1 -> ATTACK
	exp0 := math.Exp(float64(logits[0]))
	exp1 := math.Exp(float64(logits[1]))
	sum := exp0 + exp1

	// Attack probability is index 1
	attackProb := exp1 / sum

	return attackProb > 0.5
}

func (d *Detector) ExtractText(row map[string]string) string {
	path := strings.TrimSpace(row["path"])
	query := strings.TrimSpace(row["query"])

	vals := []string{}
	// Mimic typical real server lines: e.g., "/search?q=apple"
	// This grounds the model since it was trained mostly on full URLs and raw payloads
	if path != "" && path != "nan" {
		if query != "" && query != "nan" {
			vals = append(vals, path+"?"+query)
		} else {
			vals = append(vals, path)
		}
	} else if query != "" && query != "nan" {
		vals = append(vals, query)
	}

	for _, f := range []string{"headers", "body"} {
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

// Basic WordPiece Tokenizer implementation
func (d *Detector) Tokenize(text string) ([]int64, []int64) {
	cleaned := d.CleanText(text)

	// Default tokens for BERT
	clsTokenID := int64(101)
	sepTokenID := int64(102)
	padTokenID := int64(0)
	unkTokenID := int64(100)

	words := regexp.MustCompile(`\w+|[^\w\s]`).FindAllString(cleaned, -1)

	var tokens []int64
	tokens = append(tokens, clsTokenID)

	for _, word := range words {
		if len(tokens) >= d.maxLen-1 { // Reserve space for SEP
			break
		}

		start := 0
		for start < len(word) {
			end := len(word)
			var subTokenID int64 = -1
			var matchLen int

			for end > start {
				subWord := word[start:end]
				if start > 0 {
					subWord = "##" + subWord
				}

				if id, exists := d.vocab[subWord]; exists {
					subTokenID = int64(id)
					matchLen = end - start
					break
				}
				end--
			}

			if subTokenID == -1 {
				tokens = append(tokens, unkTokenID)
				break
			}

			tokens = append(tokens, subTokenID)
			start += matchLen

			if len(tokens) >= d.maxLen-1 {
				break
			}
		}
	}

	tokens = append(tokens, sepTokenID)

	inputIDs := make([]int64, d.maxLen)
	attentionMask := make([]int64, d.maxLen)

	for i := 0; i < d.maxLen; i++ {
		if i < len(tokens) {
			inputIDs[i] = tokens[i]
			attentionMask[i] = 1
		} else {
			inputIDs[i] = padTokenID
			attentionMask[i] = 0
		}
	}

	return inputIDs, attentionMask
}
