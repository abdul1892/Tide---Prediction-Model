import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np, pandas as pd, keras
from sklearn.preprocessing import MinMaxScaler

D, LOOKBACK = r"C:\Users\Abdul Rahman J\Downloads", 24
df = pd.read_excel(rf"{D}\Tidedata_6m.xlsx")
s  = df["Tide"].astype("float32").values.reshape(-1, 1)
i_tr, i_val = int(len(s) * 0.70), int(len(s) * 0.85)
scaler = MinMaxScaler().fit(s[:i_tr]); z = scaler.transform(s)
m = keras.saving.load_model(rf"{D}\tide_model_2.keras")

def resid(lo, hi=None):
    a = z[lo - LOOKBACK:hi]
    W = np.array([a[i:i + LOOKBACK + 1] for i in range(len(a) - LOOKBACK)])
    p = scaler.inverse_transform(m.predict(W[:, :-1], verbose=0)).ravel()
    y = scaler.inverse_transform(W[:, -1]).ravel()
    return y, p, y - p

_,  _,  rv = resid(i_tr, i_val)                     # validation -> fit phi here only
at, pt, _  = resid(i_val)                           # test

num = float((rv[:-1] * rv[1:]).sum())               # least-squares phi, seeded on validation
den = float((rv[:-1] ** 2).sum())
corr, phis, e_prev = np.empty_like(pt), np.empty_like(pt), 0.0
for i in range(len(pt)):
    phi = num / den
    corr[i] = pt[i] + phi * e_prev                  # correct using ONLY the previous error
    e = at[i] - pt[i]                               # the actual arrives after the prediction
    num += e_prev * e; den += e_prev ** 2           # online update
    phis[i], e_prev = phi, e

def sc(p):
    r = at - p
    return np.sqrt((r**2).mean()), np.abs(r).mean(), 1 - (r**2).sum() / ((at - at.mean())**2).sum()

(r0, m0, q0), (r1, m1, q1) = sc(pt), sc(corr)
print(f"phi final {phis[-1]:.3f}")
print(f"raw       RMSE {r0:.4f}  MAE {m0:.4f}  R2 {q0:.4f}")
print(f"corrected RMSE {r1:.4f}  MAE {m1:.4f}  R2 {q1:.4f}   ({100*(r1-r0)/r0:+.1f}%)")