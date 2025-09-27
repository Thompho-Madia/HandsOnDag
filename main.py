# fx_predictor.py
import os
import json
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
import joblib
import sqlite3

# ---------------------------
# Config
# ---------------------------
DATA_DIR = "fx_data"
MODEL_DIR = "fx_models"
CACHE_DB = "pred_cache.db"
POPULAR_PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD"]  # pre-train these daily
TIMEFRAMES = {
    "5m": {"interval": "5m", "lookback_days": 7},   # intraday short window
    "2h": {"interval": "60m", "resample": "2H", "lookback_days": 60},
    "4h": {"interval": "60m", "resample": "4H", "lookback_days": 90},
    "1d": {"interval": "1d", "lookback_days": 365},
}
SEQUENCE_LENGTH = 24   # adjust per timeframe; for example 24 steps
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Ensure SQLite cache exists
def init_cache_db():
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        pair TEXT,
        timeframe TEXT,
        pred_date TEXT,
        prediction_json TEXT,
        updated_at TEXT,
        PRIMARY KEY (pair, timeframe, pred_date)
    )
    """)
    conn.commit()
    conn.close()

init_cache_db()

# ---------------------------
# Utilities: pair to ticker
# ---------------------------
def pair_to_yf_ticker(pair: str) -> str:
    """
    Turns 'EURUSD' or 'EUR/USD' into Yahoo ticker 'EURUSD=X'
    """
    p = pair.upper().replace("/", "")
    return f"{p}=X"

# ---------------------------
# Data fetching
# ---------------------------
def fetch_ohlc(pair: str, interval: str, lookback_days: int) -> pd.DataFrame:
    """
    Fetch OHLC data using yfinance. Returns a dataframe with columns: Open, High, Low, Close, Volume (if present).
    Note: yfinance intraday data is limited to recent days (typically 7 days for 1m/2m/5m).
    For production use paid data APIs for longer intraday history.
    """
    ticker = pair_to_yf_ticker(pair)
    # yfinance expects period argument like "7d", "365d"
    period = f"{lookback_days}d"
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False)
    except Exception as e:
        raise RuntimeError(f"Data fetch failed for {pair} {interval}: {e}")

    if df.empty:
        raise RuntimeError(f"No data returned for {pair} {interval}. Check ticker or data source limits.")
    # Ensure datetime index
    df = df[['Open', 'High', 'Low', 'Close']].dropna()
    df.index = pd.to_datetime(df.index)
    return df

# ---------------------------
# Resampling helper (for 2H/4H when using 1h base)
# ---------------------------
def resample_df(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLC by rule (e.g. '2H', '4H', 'D').
    """
    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }
    df_res = df.resample(rule).agg(agg).dropna()
    return df_res

# ---------------------------
# Feature engineering
# ---------------------------
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple features: returns, high-low range, moving averages, RSI (optional via ta).
    Keep it minimal to avoid heavy dependencies.
    """
    df = df.copy()
    df['ret'] = df['Close'].pct_change().fillna(0)
    df['hl_range'] = (df['High'] - df['Low']) / df['Open']
    df['ma5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['ma10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df = df.dropna()
    return df

# ---------------------------
# Labels and sequences
# ---------------------------
def make_labels(df: pd.DataFrame) -> pd.Series:
    """
    Binary label: 1 if next step close > current close else 0.
    """
    return (df['Close'].shift(-1) > df['Close']).astype(int)[:-1]  # drop last NaN

def build_sequences(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Build sequences and labels. Returns X, y, scaler (fit on training features).
    """
    assert 'Close' in df.columns
    df_fe = add_technical_features(df)
    # Keep features
    features = ['Open', 'High', 'Low', 'Close', 'ret', 'hl_range', 'ma5', 'ma10']
    X_raw = df_fe[features].values
    y_raw = (df_fe['Close'].shift(-1) > df_fe['Close']).astype(int).values
    # Remove last row where label is NaN
    X_raw = X_raw[:-1, :]
    y_raw = y_raw[:-1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    sequences = []
    labels = []
    for i in range(seq_len, len(X_scaled)):
        sequences.append(X_scaled[i-seq_len:i, :])
        labels.append(y_raw[i])
    X = np.array(sequences)
    y = np.array(labels)
    return X, y, scaler

# ---------------------------
# PyTorch Model
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=16):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim2, output_dim)
        self.fc2 = nn.Linear(output_dim, 1)
        
    def forward(self, x):
        # LSTM layers
        lstm1_out, _ = self.lstm1(x)
        # Apply batch norm and dropout
        lstm1_out = lstm1_out.permute(0, 2, 1)  # Change shape for batch norm
        lstm1_out = self.bn1(lstm1_out)
        lstm1_out = lstm1_out.permute(0, 2, 1)  # Change back
        lstm1_out = self.dropout1(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        # Get last time step
        lstm2_out = lstm2_out[:, -1, :]
        lstm2_out = self.bn2(lstm2_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Fully connected layers
        fc1_out = F.relu(self.fc1(lstm2_out))
        output = torch.sigmoid(self.fc2(fc1_out))
        return output

def build_lstm_model(input_dim: int) -> LSTMModel:
    model = LSTMModel(input_dim=input_dim)
    return model.to(device)

# ---------------------------
# Save / Load model + scaler
# ---------------------------
def model_path(pair: str, timeframe: str) -> str:
    return os.path.join(MODEL_DIR, f"{pair}_{timeframe}.pth")

def scaler_path(pair: str, timeframe: str) -> str:
    return os.path.join(MODEL_DIR, f"{pair}_{timeframe}_scaler.pkl")

# ---------------------------
# Train pipeline
# ---------------------------
def train_for_pair_timeframe(pair: str, timeframe: str,
                             seq_len: int = SEQUENCE_LENGTH,
                             epochs: int = EPOCHS,
                             batch_size: int = BATCH_SIZE,
                             force_retrain: bool = False) -> Dict:
    """
    Fetch data, build sequences, train model, save model + scaler, return training metrics summary.
    """
    tf_cfg = TIMEFRAMES[timeframe]
    interval = tf_cfg['interval']
    lookback = tf_cfg['lookback_days']
    df = fetch_ohlc(pair, interval=interval, lookback_days=lookback)

    # optional resample if required
    if 'resample' in tf_cfg:
        df = resample_df(df, tf_cfg['resample'])

    if len(df) < seq_len + 10:
        raise RuntimeError(f"Not enough data for {pair} {timeframe}: {len(df)} rows.")

    X, y, scaler = build_sequences(df, seq_len)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device).unsqueeze(1)
    
    # Train/test split
    split_idx = int(0.85 * len(X_tensor))
    X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[2]
    model_file = model_path(pair, timeframe)
    best_model_file = model_file.replace('.pth', '.best.pth')

    model = build_lstm_model(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_file)
        
        # Early stopping
        if epoch > 5 and val_losses[-1] > val_losses[-2]:
            patience_counter += 1
            if patience_counter >= 5:
                break
        else:
            patience_counter = 0

    # Load best model and save
    model.load_state_dict(torch.load(best_model_file))
    torch.save(model.state_dict(), model_file)
    joblib.dump(scaler, scaler_path(pair, timeframe))

    # Clean up best model file
    if os.path.exists(best_model_file):
        os.remove(best_model_file)

    # return summary
    return {
        "pair": pair,
        "timeframe": timeframe,
        "val_accuracy": val_accuracies[-1] if val_accuracies else 0.0,
        "val_loss": val_losses[-1] if val_losses else float('inf'),
        "trained_at": datetime.utcnow().isoformat()
    }

# ---------------------------
# Predict pipeline
# ---------------------------
def predict_pair_timeframe(pair: str, timeframe: str, seq_len: int = SEQUENCE_LENGTH) -> Dict:
    """
    Load model+scaler, fetch most recent data, produce a prediction (probability) for next step up/down.
    Adds confidence score and risk level.
    """
    model_file = model_path(pair, timeframe)
    scaler_file = scaler_path(pair, timeframe)
    if not os.path.exists(model_file) or not os.path.exists(scaler_file):
        raise FileNotFoundError("Model or scaler not found. Train first.")

    # Load model and scaler
    tf_cfg = TIMEFRAMES[timeframe]
    df = fetch_ohlc(pair, interval=tf_cfg['interval'], lookback_days=tf_cfg['lookback_days'])
    if 'resample' in tf_cfg:
        df = resample_df(df, tf_cfg['resample'])

    df_fe = add_technical_features(df)
    features = ['Open', 'High', 'Low', 'Close', 'ret', 'hl_range', 'ma5', 'ma10']
    X_raw = df_fe[features].values
    if len(X_raw) < seq_len:
        raise RuntimeError("Not enough data to predict - need more history.")

    # Load scaler and prepare input
    scaler = joblib.load(scaler_file)
    
    # Compute volatility from recent returns
    df_fe['returns'] = df_fe['Close'].pct_change()
    volatility = df_fe['returns'].tail(50).std()  # last 50 points (tweak if needed)
    volatility = float(volatility) if not np.isnan(volatility) else 0.0

    # Prepare input sequence
    last_seq = X_raw[-seq_len:, :]
    last_seq_scaled = scaler.transform(last_seq)
    X_input = torch.FloatTensor(last_seq_scaled).unsqueeze(0).to(device)

    # Load and initialize model
    input_dim = X_input.shape[2]
    model = build_lstm_model(input_dim)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # Predict probability
    with torch.no_grad():
        prob_up = float(model(X_input).cpu().numpy()[0][0])
    
    label = int(prob_up > 0.5)

    # Confidence score
    confidence = float(max(prob_up, 1 - prob_up))

    # Risk classification based on confidence and volatility
    if confidence >= 0.8 and volatility < 0.005:
        risk = "low"
    elif confidence >= 0.65 and volatility < 0.01:
        risk = "medium"
    else:
        risk = "high"

    return {
        "pair": pair,
        "timeframe": timeframe,
        "prob_up": prob_up,
        "pred_label": label,
        "confidence": confidence,
        "volatility": volatility,
        "risk": risk,
        "timestamp_utc": datetime.utcnow().isoformat()
    }

# ---------------------------
# Cache management
# ---------------------------
def save_prediction_cache(pair: str, timeframe: str, pred: Dict):
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    pred_date = date.today().isoformat()
    c.execute("""
    INSERT OR REPLACE INTO predictions (pair, timeframe, pred_date, prediction_json, updated_at)
    VALUES (?, ?, ?, ?, ?)
    """, (pair, timeframe, pred_date, json.dumps(pred), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def load_prediction_cache(pair: str, timeframe: str) -> Optional[Dict]:
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    pred_date = date.today().isoformat()
    c.execute("""
    SELECT prediction_json FROM predictions WHERE pair=? AND timeframe=? AND pred_date=?
    """, (pair, timeframe, pred_date))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

# ---------------------------
# Popular pairs daily trainer
# ---------------------------
def train_popular_pairs():
    print(f"[{datetime.utcnow().isoformat()}] Starting daily training of popular pairs...")
    for p in POPULAR_PAIRS:
        for tf in TIMEFRAMES.keys():
            try:
                print(f"Training {p} {tf} ...")
                res = train_for_pair_timeframe(p, tf, seq_len=SEQUENCE_LENGTH, epochs=EPOCHS)
                # After training, produce and cache prediction
                pred = predict_pair_timeframe(p, tf, seq_len=SEQUENCE_LENGTH)
                save_prediction_cache(p, tf, pred)
                print(f"Done {p} {tf} -> acc {res['val_accuracy']}")
            except Exception as e:
                print(f"Failed training {p} {tf}: {e}")

# ---------------------------
# FastAPI for frontend
# ---------------------------
app = FastAPI(title="FX LSTM Predictor")

class PredictRequest(BaseModel):
    pair: str  # E.g. "EURUSD" or "EUR/USD"
    force_retrain: Optional[bool] = False
    timeframes: Optional[List[str]] = None  # optional subset

@app.on_event("startup")
def startup_tasks():
    # start scheduler for daily retrain of popular pairs at e.g. 01:00 UTC
    scheduler = BackgroundScheduler()
    # schedule at 01:00 UTC every day
    scheduler.add_job(train_popular_pairs, 'cron', hour=1, minute=0)
    scheduler.start()
    # store scheduler on app so we don't lose ref
    app.state.scheduler = scheduler
    print("Scheduler started. Also completing initial cache population for popular pairs (background).")
    # Optionally kick off an initial training run (commented to avoid heavy startup).
    # train_popular_pairs()

@app.post("/predict")
def predict(req: PredictRequest):
    pair = req.pair.upper().replace("/", "")
    timeframes = req.timeframes if req.timeframes else list(TIMEFRAMES.keys())
    response = {"pair": pair, "predictions": {}, "trained": {}}
    for tf in timeframes:
        if tf not in TIMEFRAMES:
            raise HTTPException(status_code=400, detail=f"Unsupported timeframe {tf}")
        # check cache (today)
        cached = load_prediction_cache(pair, tf)
        if cached and not req.force_retrain:
            response["predictions"][tf] = {"source": "cache", "data": cached}
            continue
        # if not cached and pair is popular, attempt to use pre-trained model file
        model_file = model_path(pair, tf)
        try:
            if os.path.exists(model_file) and not req.force_retrain:
                pred = predict_pair_timeframe(pair, tf)
                save_prediction_cache(pair, tf, pred)
                response["predictions"][tf] = {"source": "model_file", "data": pred}
            else:
                # train on the pair now (may take time)
                train_res = train_for_pair_timeframe(pair, tf, seq_len=SEQUENCE_LENGTH, epochs=EPOCHS)
                response["trained"][tf] = train_res
                pred = predict_pair_timeframe(pair, tf)
                save_prediction_cache(pair, tf, pred)
                response["predictions"][tf] = {"source": "trained_now", "data": pred}
        except Exception as e:
            response["predictions"][tf] = {"error": str(e)}
    return response

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# Optional: endpoint to force retrain popular pairs now
@app.post("/admin/retrain_popular")
def retrain_popular():
    try:
        train_popular_pairs()
        return {"status": "ok", "message": "Popular pairs retrained"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# If running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
