import pandas as pd
import numpy as np
from datetime import datetime

# ---- Dummy Data Generator ----
def make_dummy_data(rows=20):
    """Generate dummy OHLCV data with required columns."""
    data = {
        "Open": np.random.uniform(1.0, 1.5, rows),
        "High": np.random.uniform(1.5, 2.0, rows),
        "Low": np.random.uniform(0.5, 1.0, rows),
        "Close": np.random.uniform(1.0, 1.5, rows),
        "Volume": np.random.randint(1000, 5000, rows),
    }
    return pd.DataFrame(data)

# ---- Health Check ----
print("==== Health Check ====")
print(f"Status: OK | Time: {datetime.utcnow().isoformat()}\n")

# ---- Multiple Predictions Test ----
test_cases = [
    ("EURUSD", ["1d"]),
    ("GBPUSD", ["4h"]),
    ("USDJPY", ["2h"]),
    ("AUDUSD", ["5m"]),
    ("USDCAD", ["1d", "4h"]),
]

print("==== Multiple Predictions Test ====")
for pair, timeframes in test_cases:
    print(f"\n--- Test Case: {pair} ({', '.join(timeframes)}) ---")
    
    for tf in timeframes:
        try:
            df = make_dummy_data()
            # Fake prediction
            prob_up = np.random.rand()
            confidence = max(prob_up, 1 - prob_up)
            pred_label = "UP" if prob_up > 0.5 else "DOWN"
            risk_level = "High" if confidence < 0.6 else "Medium"
            volatility = df["Close"].pct_change().std()

            # Print nicely
            print(f"Timeframe: {tf}")
            print(f"  Predicted Trend : {pred_label}")
            print(f"  Probability Up  : {prob_up:.2f}")
            print(f"  Confidence      : {confidence:.2f}")
            print(f"  Risk Level      : {risk_level}")
            print(f"  Volatility      : {volatility:.5f}")
            print(f"  Timestamp (UTC) : {datetime.utcnow().isoformat()}")
            print("-" * 40)

        except Exception as e:
            print(f"  ERROR for {pair} {tf}: {e}")
            print("-" * 40)
