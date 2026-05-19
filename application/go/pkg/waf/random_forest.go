package waf

var _ Model = (*RandomForestModel)(nil)

// RandomForestModel classifies HTTP requests using a Random Forest ONNX model.
type RandomForestModel struct {
	baseModel
}

func newRandomForestModel(sharedLibPath string, cfg ModelConfig) (*RandomForestModel, error) {
	base, err := newBaseModel(cfg, sharedLibPath, denseEncoder{})
	if err != nil {
		return nil, err
	}
	return &RandomForestModel{baseModel: *base}, nil
}
