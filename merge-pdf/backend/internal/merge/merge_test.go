package merge

import (
	"testing"

	"github.com/ml/merge-pdf/backend/internal/model"
)

func TestSortInputs(t *testing.T) {
	inputs := []model.MergeFileInput{
		{Name: "10.pdf", Order: 10},
		{Name: "2-b.pdf", Order: 2},
		{Name: "2-a.pdf", Order: 2},
	}

	SortInputs(inputs)

	want := []string{"2-a.pdf", "2-b.pdf", "10.pdf"}
	for i, item := range inputs {
		if item.Name != want[i] {
			t.Fatalf("position %d got %s want %s", i, item.Name, want[i])
		}
	}
}
