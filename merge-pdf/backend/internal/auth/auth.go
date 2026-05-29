package auth

import (
	"fmt"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"

	"github.com/ml/merge-pdf/backend/internal/model"
)

type Claims struct {
	UserID int64      `json:"userId"`
	Role   model.Role `json:"role"`
	Email  string     `json:"email"`
	jwt.RegisteredClaims
}

type Service struct {
	secret []byte
}

// NewService builds the shared auth helper once so every handler signs and validates tokens consistently.
func NewService(secret string) Service {
	return Service{secret: []byte(secret)}
}

// HashPassword keeps plaintext passwords out of storage before they ever reach the database.
func (s Service) HashPassword(password string) (string, error) {
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return "", fmt.Errorf("hash password: %w", err)
	}
	return string(hash), nil
}

// CheckPassword compares a login attempt against the stored bcrypt hash.
func (s Service) CheckPassword(hash, password string) error {
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
}

// GenerateToken issues a short-lived bearer token that frontend requests can reuse cheaply.
func (s Service) GenerateToken(user model.User) (string, error) {
	now := time.Now()
	claims := Claims{
		UserID: user.ID,
		Role:   user.Role,
		Email:  user.Email,
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   fmt.Sprintf("%d", user.ID),
			IssuedAt:  jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(now.Add(24 * time.Hour)),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	signed, err := token.SignedString(s.secret)
	if err != nil {
		return "", fmt.Errorf("sign token: %w", err)
	}
	return signed, nil
}

// ParseToken validates a bearer token before protected handlers trust its identity claims.
func (s Service) ParseToken(tokenString string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (any, error) {
		return s.secret, nil
	})
	if err != nil {
		return nil, fmt.Errorf("parse token: %w", err)
	}

	claims, ok := token.Claims.(*Claims)
	if !ok || !token.Valid {
		return nil, fmt.Errorf("invalid token claims")
	}

	return claims, nil
}

// CanAccessJob keeps history data scoped to owners while still allowing admin support access.
func CanAccessJob(actor model.User, jobUserID int64) bool {
	return actor.Role == model.RoleAdmin || actor.ID == jobUserID
}
