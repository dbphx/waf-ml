package waf

import (
	"sync"
	"time"
)

// Detector is an interface that both Random Forest and Logistic Regression detectors must implement.
// This allows the ReputationManager to work with either model interchangeably.
type Detector interface {
	// PredictScore returns the raw probability score (0.0 to 1.0) of a request being an attack.
	PredictScore(args map[string]string) (float64, error)
	// Predict returns a boolean decision based on internal threshold (legacy support).
	Predict(args map[string]string) bool
}

// ClientState holds the reputation data for a single IP address.
type ClientState struct {
	ReputationScore float64
	LastSeen        time.Time
}

// ReputationManager handles stateful tracking of client IP addresses.
// It wraps a stateless ML Detector to add memory and behavior analysis.
type ReputationManager struct {
	detector Detector

	// State Storage
	reputation map[string]*ClientState
	mu         sync.RWMutex

	// Configuration
	BlockThreshold     float64       // Score above this is blocked (e.g., 0.80)
	SuspicionThreshold float64       // Score above this increases reputation (e.g., 0.50)
	TTL                time.Duration // How long to remember a client (e.g., 24 hours)
	DecayFactor        float64       // Multiplier for score accumulation (e.g., 0.1 adds 10% of score)
}

// NewReputationManager creates a new stateful WAF manager.
func NewReputationManager(d Detector, blockThreshold, suspicionThreshold float64, ttl time.Duration) *ReputationManager {
	return &ReputationManager{
		detector:           d,
		reputation:         make(map[string]*ClientState),
		BlockThreshold:     blockThreshold,
		SuspicionThreshold: suspicionThreshold,
		TTL:                ttl,
		DecayFactor:        0.1, // Default: add 10% of the suspicious score to reputation
	}
}

// AnalyzeRequest processes a request from a specific client IP.
// It returns:
// - blocked: true if the request should be blocked
// - finalScore: the combined score (Model + Reputation)
// - details: a string explaining the decision
func (rm *ReputationManager) AnalyzeRequest(clientIP string, requestArgs map[string]string) (bool, float64, string) {
	// 1. Get Stateless Score from ML Model
	modelScore, err := rm.detector.PredictScore(requestArgs)
	if err != nil {
		// Fail open on model error, or handle as configured
		return false, 0.0, "Model Error"
	}

	// 2. Retrieve Current Reputation
	rm.mu.RLock()
	state, exists := rm.reputation[clientIP]
	rm.mu.RUnlock()

	currentReputation := 0.0

	// 3. Check TTL (Expiration)
	if exists {
		if time.Since(state.LastSeen) > rm.TTL {
			// Expired: Treat as new client (score 0)
			// We will update/overwrite this entry later if needed
		} else {
			currentReputation = state.ReputationScore
		}
	}

	// 4. Calculate Final Score
	// Formula: Final = Model_Confidence + Past_Reputation
	finalScore := modelScore + currentReputation

	// 5. Update Reputation if suspicious
	// Only update state if the model found something suspicious OR they already have a bad reputation
	if modelScore > rm.SuspicionThreshold || currentReputation > 0 {
		rm.mu.Lock()

		// Re-read state in case of race condition during lock upgrade
		state, exists = rm.reputation[clientIP]

		newReputation := currentReputation

		// If this specific request was suspicious, increase the reputation
		if modelScore > rm.SuspicionThreshold {
			// Add a fraction of the model score to the long-term reputation
			// Example: Score 0.6 -> adds 0.06 to reputation
			newReputation += (modelScore * rm.DecayFactor)
		}

		// Cap reputation to avoid overflow (optional, e.g., max 2.0)
		if newReputation > 2.0 {
			newReputation = 2.0
		}

		rm.reputation[clientIP] = &ClientState{
			ReputationScore: newReputation,
			LastSeen:        time.Now(),
		}
		rm.mu.Unlock()
	}

	// 6. Decision
	blocked := finalScore >= rm.BlockThreshold

	reason := "Clean"
	if blocked {
		reason = "Blocked (Threshold Reached)"
		if currentReputation > 0 {
			reason += " [Reputation Contributing]"
		}
	} else if finalScore > rm.SuspicionThreshold {
		reason = "Suspicious (Monitored)"
	}

	return blocked, finalScore, reason
}

// ResetIP manually clears the reputation for an IP (e.g., admin unblock)
func (rm *ReputationManager) ResetIP(clientIP string) {
	rm.mu.Lock()
	delete(rm.reputation, clientIP)
	rm.mu.Unlock()
}

// CleanupExpired removes old entries to prevent memory leaks
func (rm *ReputationManager) CleanupExpired() {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	now := time.Now()
	for ip, state := range rm.reputation {
		if now.Sub(state.LastSeen) > rm.TTL {
			delete(rm.reputation, ip)
		}
	}
}
