package auth

import (
	"testing"

	"github.com/ml/merge-pdf/backend/internal/model"
)

func TestCanAccessJob(t *testing.T) {
	user := model.User{ID: 10, Role: model.RoleUser}
	admin := model.User{ID: 11, Role: model.RoleAdmin}

	if !CanAccessJob(user, 10) {
		t.Fatalf("expected owner to access own job")
	}
	if CanAccessJob(user, 12) {
		t.Fatalf("expected non-owner user to be rejected")
	}
	if !CanAccessJob(admin, 12) {
		t.Fatalf("expected admin to access any job")
	}
}
