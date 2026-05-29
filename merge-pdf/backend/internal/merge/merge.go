package merge

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"

	pdfapi "github.com/pdfcpu/pdfcpu/pkg/api"

	"github.com/ml/merge-pdf/backend/internal/model"
)

// SortInputs keeps merge order deterministic when users or Drive provide duplicate positions.
func SortInputs(inputs []model.MergeFileInput) {
	sort.Slice(inputs, func(i, j int) bool {
		if inputs[i].Order == inputs[j].Order {
			return inputs[i].Name < inputs[j].Name
		}
		return inputs[i].Order < inputs[j].Order
	})
}

// MergeFiles delegates PDF assembly to pdfcpu so the backend stays in-process and script-free.
func MergeFiles(workDir, outputName string, inputs []model.MergeFileInput) (string, error) {
	if len(inputs) == 0 {
		return "", fmt.Errorf("no files to merge")
	}

	SortInputs(inputs)

	paths := make([]string, 0, len(inputs))
	for _, input := range inputs {
		paths = append(paths, input.LocalPath)
	}

	outputPath := filepath.Join(workDir, outputName)
	if err := pdfapi.MergeCreateFile(paths, outputPath, false, nil); err != nil {
		return "", fmt.Errorf("merge pdf files: %w", err)
	}

	if _, err := os.Stat(outputPath); err != nil {
		return "", fmt.Errorf("merged output missing: %w", err)
	}

	return outputPath, nil
}
