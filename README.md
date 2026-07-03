# Candlestick Pattern Recognition with GAF-CNN-LSTM

A deep learning pipeline that detects classical Japanese candlestick reversal patterns in NIFTY 50 minute-level OHLC data by encoding price sequences as Gramian Angular Field (GAF) images and classifying them with a CNN-LSTM network.

## Overview

Rule-based logic first labels candlestick patterns in historical price data. Each labeled candle is then converted into a short sequence of multi-channel GAF images, which preserve the temporal and structural characteristics of price action in image form. A time-distributed CNN extracts spatial features from each image in the sequence, and an LSTM layer models the temporal relationship between them, allowing the model to classify the candlestick pattern present at the end of the sequence.

## Pipeline

1. **Data ingestion** — Load minute-level OHLC data for NIFTY 50 from `nifty50.csv`.
2. **Feature engineering** — Derive candle body size, upper shadow, and lower shadow from OHLC values.
3. **Trend detection** — Classify local trend (uptrend / downtrend / sideways) using a rolling moving average.
4. **Pattern labeling** — Apply rule-based detectors for six candlestick patterns:
   - Morning Star
   - Evening Star
   - Bullish Engulfing
   - Bearish Engulfing
   - Bullish Harami
   - Bearish Harami
5. **GAF encoding** — Transform sliding windows of price/feature series into Gramian Angular Field images (4 channels: Close, Body, Lower Shadow, Upper Shadow), grouped into short sequences.
6. **Model training** — Train a CNN-LSTM classifier (with focal loss and class balancing) to predict the pattern label from a GAF image sequence.
7. **Evaluation** — Assess performance with a confusion matrix and classification report.

## Model Architecture

- **CNN block** (applied per time step via `TimeDistributed`): two convolutional layers with max pooling, ReLU activation, and dropout, followed by flattening.
- **LSTM layer**: processes the sequence of CNN feature vectors to capture temporal dependencies across the window.
- **Dense head**: fully connected layer followed by a softmax output over pattern classes.
- **Loss function**: categorical focal loss, used to handle class imbalance among pattern types.
- **Optimizer**: Adam.

## Repository Contents

| File | Description |
|---|---|
| `candlestick1.ipynb` | End-to-end notebook: data loading, feature engineering, pattern labeling, GAF encoding, model training, and evaluation |
| `nifty50.csv` | Minute-level OHLC price data for the NIFTY 50 index |
| `requirements.txt` | Python dependencies |

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Open and run the notebook:

```bash
jupyter notebook candlestick1.ipynb
```

Run the cells sequentially to reproduce the full pipeline, from raw data to trained model and evaluation metrics.

## Data

The dataset (`nifty50.csv`) contains minute-level OHLC records for the NIFTY 50 index with the following columns:

| Column | Description |
|---|---|
| `index` | Instrument identifier |
| `date` | Trading date (YYYYMMDD) |
| `time` | Trading time (HH:MM) |
| `open` | Opening price |
| `high` | Highest price |
| `low` | Lowest price |
| `close` | Closing price |

## Evaluation

Model performance is reported using:

- **Confusion matrix** across all six pattern classes.
- **Classification report** with per-class precision, recall, and F1-score.

## Notes and Limitations

- Pattern labels are generated using deterministic technical-analysis rules, not manually verified ground truth, so labeling accuracy is bounded by the quality of those rules.
- The model is trained on a single index (NIFTY 50) and has not been validated on other instruments or timeframes.
- Class imbalance across pattern types is mitigated using computed class weights and a focal loss function, but the "None" class is excluded from the classification task entirely, meaning the model only distinguishes between confirmed pattern types.
- This project is intended for research and educational purposes and does not constitute financial or trading advice.

## License

Specify a license for this project (e.g., MIT, Apache 2.0) before distribution.
