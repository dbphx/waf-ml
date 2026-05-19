package waf

var _ Model = (*LogisticRegressionModel)(nil)

// LogisticRegressionModel classifies HTTP requests using a Logistic Regression ONNX model.
type LogisticRegressionModel struct {
	baseModel
}

func newLogisticRegressionModel(sharedLibPath string, cfg ModelConfig) (*LogisticRegressionModel, error) {
	base, err := newBaseModel(cfg, sharedLibPath, denseEncoder{})
	if err != nil {
		return nil, err
	}
	return &LogisticRegressionModel{baseModel: *base}, nil
}
