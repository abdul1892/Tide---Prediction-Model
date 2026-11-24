*Tide-Level Forecasting with LSTM Networks*

Short-horizon tide forecasting that learns directly from a station's own historical tide-gauge readings, using stacked LSTM networks on raw water level — no harmonic decomposition, no external inputs.

Harmonic analysis, the standard method, needs years of continuous record to resolve its constituents and models only the astronomical tide, so it cannot respond to wind or pressure effects. A recurrent network learns from a short record instead, and fits whatever structure the observed signal actually contains.

The repository compares three LSTM depths under one fixed protocol, searches layer width by random search with repeated seeds so that real gains can be separated from seed noise, scores saved models through a standalone evaluator that keeps the test set out of every tuning decision, measures everything against a naive persistence baseline, and applies an online residual correction that improves forecasts without retraining the network.
