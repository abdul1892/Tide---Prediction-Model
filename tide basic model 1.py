import os
import time
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np, pandas as pd, keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
keras.utils.set_random_seed(42)
# --- Load raw data (univariate tide height) ---
df = pd.read_excel(r"C:\Users\Abdul Rahman J\Downloads\Tidedata_6m.xlsx")
series = df["Tide"].astype("float32").values.reshape(-1, 1)
# --- Chronological 70/15/15 split; scaler fit on TRAIN only (no leakage) ---
n = len(series)
i_tr = int(n * 0.70); i_val = int(n * 0.85)
scaler = MinMaxScaler().fit(series[:i_tr])
scaled = scaler.transform(series)
# --- Windowing: 24 past steps (12 h) -> predict next step ---
LOOKBACK = 24
def make_windows(a):
    X, y = [], []
    for i in range(len(a) - LOOKBACK):
        X.append(a[i:i + LOOKBACK]); y.append(a[i + LOOKBACK])
    return np.array(X), np.array(y)
X_train, y_train = make_windows(scaled[:i_tr])
X_val,   y_val   = make_windows(scaled[i_tr - LOOKBACK:i_val])
X_test,  y_test  = make_windows(scaled[i_val - LOOKBACK:])
# --- Model: 1 hidden LSTM layer (64, relu) + dropout 0.2 -> Dense(1) ---
model = keras.Sequential([
    layers.Input((LOOKBACK, 1)),
    layers.LSTM(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")
# --- Print loss every 10 epochs to watch what the model is doing ---
class Every10(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 10 == 0: self.block_start = time.time()
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            block = time.time() - self.block_start
            print(f"epoch {epoch+1:3d}  train_loss {logs['loss']:.5f}  val_loss {logs['val_loss']:.5f}  last_10_epochs {block:.2f}s")

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=60, batch_size=64, verbose=0, callbacks=[Every10()])
# --- Metrics on all three sets (inverse-scaled to metres) ---
def report(name, X, y):
    p = scaler.inverse_transform(model.predict(X, verbose=0)); t = scaler.inverse_transform(y)
    print(f"{name:5s} RMSE {np.sqrt(mean_squared_error(t, p)):.4f} m   MAE {mean_absolute_error(t, p):.4f} m   R2 {r2_score(t, p):.4f}")
    return t, p
report("Train", X_train, y_train)
report("Val",   X_val,   y_val)
true, pred = report("Test", X_test, y_test)

# --- Save trained model + scaler (before plots, so it always runs) ---
import joblib
model.save(r"C:\Users\Abdul Rahman J\Downloads\tide_model_1.keras")
joblib.dump(scaler, r"C:\Users\Abdul Rahman J\Downloads\tide_scaler.pkl")
print("saved model 1 + scaler")

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.legend(); plt.title("Learning curves")
plt.savefig(r"C:\Users\Abdul Rahman J\Downloads\learning_curves.png", dpi=120); plt.show()

plt.figure()
plt.plot(true[:150], label="actual")
plt.plot(pred[:150], label="predicted")
plt.xlabel("test step (30-min)"); plt.ylabel("tide (m)")
plt.legend(); plt.title("Test set: predicted vs actual (first ~3 days)")
plt.savefig(r"C:\Users\Abdul Rahman J\Downloads\pred_vs_actual.png", dpi=120); plt.show()