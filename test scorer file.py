"""Score already-trained .keras models on train / val / test. Inference only - no training.

  python "score_saved_models.py" sweep1_u132_s0.keras
  python "score_saved_models.py" "sweep1_u132_s*.keras"        (glob, all 3 seeds)
  python "score_saved_models.py" "sweep2_u*.keras"             (a whole sweep)
  python "score_saved_models.py"                               (defaults to sweep1_u*)
"""
import os, sys, glob
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np, pandas as pd, keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score

LOOKBACK = 24
D = r"C:\Users\Abdul Rahman J\Downloads"
OUT = os.path.join(D, "scored_models.csv")

s = pd.read_excel(os.path.join(D, "Tidedata_6m.xlsx"))["Tide"].astype("float32").values.reshape(-1, 1)
i_tr, i_val = int(len(s) * 0.70), int(len(s) * 0.85)
scaler = MinMaxScaler().fit(s[:i_tr]); z = scaler.transform(s)      # same split + scaler as training

def win(a):
    W = np.array([a[i:i + LOOKBACK + 1] for i in range(len(a) - LOOKBACK)]); return W[:, :-1], W[:, -1]
SETS = {"train": win(z[:i_tr]), "val": win(z[i_tr - LOOKBACK:i_val]), "test": win(z[i_val - LOOKBACK:])}

pats = sys.argv[1:] or ["sweep1_u*.keras"]
files = sorted({f for p in pats for f in glob.glob(p if os.path.isabs(p) else os.path.join(D, p))})
if not files: sys.exit(f"no .keras files matched {pats} in {D}")
print(f"{len(files)} model(s)")

rows = []
for f in files:
    m = keras.saving.load_model(f)
    r = {"model": os.path.basename(f)}
    for k, (X, y) in SETS.items():
        p, t = scaler.inverse_transform(m.predict(X, verbose=0)), scaler.inverse_transform(y)
        r[k+"_rmse"], r[k+"_mae"], r[k+"_r2"] = float(np.sqrt(mse(t, p))), float(mae(t, p)), float(r2_score(t, p))
    rows.append(r); print(f"{r['model']:28s} test_rmse={r['test_rmse']:.4f}  test_r2={r['test_r2']:.4f}")
    keras.backend.clear_session()

out = pd.DataFrame(rows); out.to_csv(OUT, index=False)
print(out.to_string(index=False))
print("saved ->", OUT)
