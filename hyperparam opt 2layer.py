import os,time
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np, pandas as pd, keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
torch.set_num_threads(12)

B1, N_TRIALS = (50, 150), 10                  # tied width for BOTH layers, trials
SEEDS, LOOKBACK, LR , BATCH, MAX_EPOCHS = [0,1,2], 24, 1e-4, 64, 60
LOG = r"C:\Users\Abdul Rahman J\Downloads\random_search_units_2layer.csv"
MDIR = r"C:\Users\Abdul Rahman J\Downloads"

rng = np.random.default_rng(0)                  # same draws as the 1-layer run
CONFIGS = [64] + [int(rng.integers(*B1, endpoint=True)) for _ in range(N_TRIALS)]  # 64 = reference

series = pd.read_excel(r"C:\Users\Abdul Rahman J\Downloads\Tidedata_6m.xlsx")["Tide"].astype("float32").values.reshape(-1, 1)
i_tr, i_val = int(len(series) * 0.70), int(len(series) * 0.85)
scaler = MinMaxScaler().fit(series[:i_tr])
scaled = scaler.transform(series)

def windows(a):
    W = np.array([a[i:i + LOOKBACK + 1] for i in range(len(a) - LOOKBACK)])
    return W[:, :-1], W[:, -1]

X_tr, y_tr = windows(scaled[:i_tr])
X_va, y_va = windows(scaled[i_tr - LOOKBACK:i_val])      # test set deliberately never built

def rmse(m, X, y):
    p = scaler.inverse_transform(m.predict(X, verbose=0))
    return float(np.sqrt(mean_squared_error(scaler.inverse_transform(y), p)))

def run(u, seed):
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([
        layers.Input((LOOKBACK, 1)),
        layers.LSTM(u, activation="relu", return_sequences=True), layers.Dropout(0.2),
        layers.LSTM(u, activation="tanh"), layers.Dropout(0.3), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    t0 = time.time()
    h = m.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=MAX_EPOCHS,
              batch_size=BATCH, verbose=0, callbacks=[es])
    tr, va = rmse(m, X_tr, y_tr), rmse(m, X_va, y_va)
    m.save(rf"{MDIR}\sweep2_u{u}_s{seed}.keras")
    keras.backend.clear_session()
    return tr, va, len(h.history["loss"]), time.time() - t0

rows = []
for u in CONFIGS:
    trs, vas, ep, sec = zip(*[run(u, s) for s in SEEDS])
    rows.append({"units": u, "train_rmse": float(np.median(trs)), "val_rmse": float(np.median(vas)),
                 "seed_spread": max(vas) - min(vas), "epochs": int(np.median(ep)),
                 "seconds": round(sum(sec), 1)})
    print(rows[-1]); pd.DataFrame(rows).to_csv(LOG, index=False)

res = pd.DataFrame(rows)
base = res.loc[res.units == 64, "val_rmse"].iloc[0]
res["delta_vs_default"] = res.val_rmse - base
res["pct_change"] = 100 * res.delta_vs_default / base
res = res.sort_values("val_rmse").reset_index(drop=True)
res.to_csv(LOG, index=False); print(res.to_string(index=False))
print(f"total search time: {res.seconds.sum()/60:.1f} min")

lo, hi = B1
mt = res.units.head(5).mean()
print(f"units: top-5 mean {mt:.0f} in [{lo},{hi}] -> "
      f"{'extend DOWN' if mt < lo+.2*(hi-lo) else 'extend UP' if mt > hi-.2*(hi-lo) else 'range OK'}")
