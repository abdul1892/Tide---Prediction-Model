import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np, pandas as pd, keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates

D, LOOKBACK = r"C:\Users\Abdul Rahman J\Downloads", 24
MODEL, LABEL = "sweep1_u135_s0", "1-layer sweep  ·  LSTM(135)"
# MODEL, LABEL = "tide_model_2", "basic model 2  ·  LSTM(64) + LSTM(64)"
OUT = rf"{D}\plot_split_{MODEL}.png"
BLUE, AMBER, INK = "#1f6fb2", "#e07b39", "#5a5a57"

df = pd.read_excel(rf"{D}\Tidedata_6m.xlsx")
s  = df["Tide"].astype("float32").values.reshape(-1, 1)
ts = pd.to_datetime(df["Date&Time"])
i_tr, i_val = int(len(s) * 0.70), int(len(s) * 0.85)
scaler = MinMaxScaler().fit(s[:i_tr]); z = scaler.transform(s)
m = keras.saving.load_model(rf"{D}\{MODEL}.keras")

def resid(lo, hi=None):
    a = z[lo - LOOKBACK:hi]
    W = np.array([a[i:i + LOOKBACK + 1] for i in range(len(a) - LOOKBACK)])
    p = scaler.inverse_transform(m.predict(W[:, :-1], verbose=0)).ravel()
    y = scaler.inverse_transform(W[:, -1]).ravel()
    return y, p, y - p

_,  _,  rv = resid(i_tr, i_val)                      # validation seeds phi
at, pt, _  = resid(i_val)                            # test
num, den = float((rv[:-1] * rv[1:]).sum()), float((rv[:-1] ** 2).sum())
corr, e_prev = np.empty_like(pt), 0.0
for i in range(len(pt)):
    phi = num / den
    corr[i] = pt[i] + phi * e_prev                   # only the PREVIOUS error
    e = at[i] - pt[i]
    num += e_prev * e; den += e_prev ** 2
    e_prev = e

t = ts.iloc[i_val:i_val + len(at)].values
e_raw, e_cor = at - pt, at - corr
rr, rc = float(np.sqrt((e_raw**2).mean())), float(np.sqrt((e_cor**2).mean()))
mr, mc = float(np.abs(e_raw).mean()), float(np.abs(e_cor).mean())
lim = 1.08 * max(np.abs(e_raw).max(), np.abs(e_cor).max())

fig, ax = plt.subplots(2, 1, figsize=(12, 7.4), sharex=True, sharey=True)
for a, e, c, r, mae, name in ((ax[0], e_raw, BLUE,  rr, mr, "without penalization"),
                              (ax[1], e_cor, AMBER, rc, mc, "with penalization")):
    a.axhspan(-r, r, color=c, alpha=.09, lw=0, zorder=1)                 # +-RMSE band
    a.plot(t, e, color=c, lw=.9, zorder=2)
    a.axhline(0, color="#3a3a38", lw=1.2, zorder=3)
    a.set_ylim(-lim, lim); a.set_ylabel("prediction error  (m)")
    a.grid(axis="y", color="#e8e8e6", lw=.8); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): a.spines[sp].set_color("#c9c9c5")
    a.tick_params(colors=INK, labelsize=9)
    a.text(.006, .955, name, transform=a.transAxes, fontsize=11, fontweight="semibold", color=c, va="top")
    a.text(.006, .875, f"RMSE {r:.4f} m     MAE {mae:.4f} m     shaded = ±RMSE",
           transform=a.transAxes, fontsize=9, color=INK, va="top")

ax[0].set_title(f"Prediction error with and without the penalization loop — {LABEL}",
                fontsize=12, loc="left", pad=38)
ax[0].text(0, 1.085, f"full test set  ·  {len(at):,} points  ·  {pd.Timestamp(t[0]):%d %b} to "
           f"{pd.Timestamp(t[-1]):%d %b %Y}  ·  error = observed − predicted  ·  identical axes",
           transform=ax[0].transAxes, fontsize=9.5, color=INK)
ax[0].text(0, 1.022, f"RMSE  {rr:.4f} m  →  {rc:.4f} m   ({100*(rc-rr)/rr:+.1f}%)"
           f"      MAE  {mr:.4f} m  →  {mc:.4f} m   ({100*(mc-mr)/mr:+.1f}%)"
           f"      phi {phi:.3f}", transform=ax[0].transAxes, fontsize=9.5, color=INK)
ax[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax[1].xaxis.set_minor_locator(mdates.DayLocator())
fig.tight_layout(rect=[0, 0, .97, 1])
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
print(f"phi {phi:.3f}   raw {rr:.4f} -> corrected {rc:.4f}  ({100*(rc-rr)/rr:+.1f}%)")
print("saved ->", OUT)