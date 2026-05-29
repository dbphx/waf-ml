package drive

import "testing"

func TestExtractFolderID(t *testing.T) {
	tests := []struct {
		name string
		url  string
		want string
	}{
		{
			name: "folder path",
			url:  "https://drive.google.com/drive/u/0/folders/1s_Fyt-_KFr9X2jQpZ9UmpUOY3eIJSDn4",
			want: "1s_Fyt-_KFr9X2jQpZ9UmpUOY3eIJSDn4",
		},
		{
			name: "query id",
			url:  "https://drive.google.com/open?id=abc123",
			want: "abc123",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ExtractFolderID(tc.url)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestExtractOrder(t *testing.T) {
	tests := []struct {
		name     string
		filename string
		want     int
		wantErr  bool
	}{
		{name: "leading number", filename: "01-intro.pdf", want: 1},
		{name: "embedded number", filename: "chapter-12-final.pdf", want: 12},
		{name: "no number", filename: "appendix.pdf", wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ExtractOrder(tc.filename)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %d want %d", got, tc.want)
			}
		})
	}
}
