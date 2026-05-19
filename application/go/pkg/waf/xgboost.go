package waf

var _ Model = (*XGBoostModel)(nil)

// XGBoostModel classifies HTTP requests using an XGBoost ONNX model.
// Zero-valued features are encoded as NaN to match sparse training semantics.
type XGBoostModel struct {
	baseModel
}

func newXGBoostModel(sharedLibPath string, cfg ModelConfig) (*XGBoostModel, error) {
	base, err := newBaseModel(cfg, sharedLibPath, xgboostEncoder{})
	if err != nil {
		return nil, err
	}
	return &XGBoostModel{baseModel: *base}, nil
}
