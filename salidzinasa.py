# import numpy as np
# import matplotlib.pyplot as plt

# # =========================

# # =========================
# REAL_FILE = "./data/real_speed.txt"   # īstie dati (no OCR / GPS)
# PRED_FILE = "./data/test.txt"         # tava modeļa rezultāts

# # =========================

# # =========================
# def load_data(path):
#     with open(path, "r") as f:
#         data = [float(line.strip()) for line in f if line.strip() != ""]
#     return data

# real = load_data(REAL_FILE)
# pred = load_data(PRED_FILE)

# # =========================

# # =========================
# min_len = min(len(real), len(pred))
# real = real[:min_len]
# pred = pred[:min_len]

# print(f"Loaded {min_len} samples")

# # =========================

# # =========================
# real_np = np.array(real)
# pred_np = np.array(pred)

# mae = np.mean(np.abs(real_np - pred_np))
# mse = np.mean((real_np - pred_np) ** 2)
# rmse = np.sqrt(mse)

# print(f"MAE  (vidējā absolūtā kļūda): {mae:.2f}")
# print(f"MSE  (vidējā kvadrātiskā kļūda): {mse:.2f}")
# print(f"RMSE (kvadrātsakne no MSE): {rmse:.2f}")

# # =========================

# # =========================
# mean_real = np.mean(real_np)
# mean_pred = np.mean(pred_np)
# mean_error = np.mean(real_np - pred_np)

# print(f"Vidējais reālais ātrums: {mean_real:.2f} km/h")
# print(f"Vidējais prognozētais ātrums: {mean_pred:.2f} km/h")
# print(f"Vidējā kļūda: {mean_error:.2f} km/h")
# # =========================

# # =========================
# plt.figure()

# plt.plot(real, label="Reālais ātrums")
# plt.plot(pred, label="Prognozētais ātrums")

# plt.xlabel("Kadrs")
# plt.ylabel("Ātrums (km/h)")
# plt.title("Reālā un prognozētā ātruma salīdzinājums")

# plt.legend()
# plt.grid()

# plt.savefig("salidzinajums.png")
# plt.show()
import numpy as np
import matplotlib.pyplot as plt


# =========================
REAL_FILE = "./data/real_speed.txt"   # īstie dati
PRED_FILE = "./data/test.txt"         # modeļa prognozes


# =========================
def load_data(path):
    with open(path, "r") as f:
        data = [float(line.strip()) for line in f if line.strip() != ""]
    return data

real = load_data(REAL_FILE)
pred = load_data(PRED_FILE)


# =========================
min_len = min(len(real), len(pred))

real = real[:min_len]
pred = pred[:min_len]

print(f"Loaded {min_len} samples")


# =========================
real_np = np.array(real)
pred_np = np.array(pred)

# =========================
mae = np.mean(np.abs(real_np - pred_np))
mse = np.mean((real_np - pred_np) ** 2)
rmse = np.sqrt(mse)

print(f"MAE  (vidējā absolūtā kļūda): {mae:.2f}")
print(f"MSE  (vidējā kvadrātiskā kļūda): {mse:.2f}")
print(f"RMSE (kvadrātsakne no MSE): {rmse:.2f}")


# =========================
mean_real = np.mean(real_np)
mean_pred = np.mean(pred_np)
mean_error = np.mean(real_np - pred_np)

print(f"Vidējais reālais ātrums: {mean_real:.2f} km/h")
print(f"Vidējais prognozētais ātrums: {mean_pred:.2f} km/h")
print(f"Vidējā kļūda: {mean_error:.2f} km/h")

# =========================

# =========================
plt.figure(figsize=(12, 6))


plt.plot(
    real,
    label="Reālais ātrums",
    linewidth=2
)


plt.plot(
    pred,
    label="Prognozētais ātrums",
    linewidth=2
)


plt.xlabel("Kadrs")
plt.ylabel("Ātrums (km/h)")

plt.title("Reālā un prognozētā ātruma salīdzinājums")

# =========================

# =========================
metrics_text = (
    f"MAE:  {mae:.2f}\n"
    f"MSE:  {mse:.2f}\n"
    f"RMSE: {rmse:.2f}\n\n"
    f"Vid. reālais: {mean_real:.2f} km/h\n"
    f"Vid. prognozētais: {mean_pred:.2f} km/h\n"
    f"Vid. kļūda: {mean_error:.2f} km/h"
)

plt.text(
    0.02,
    0.98,
    metrics_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)


plt.legend()


plt.grid(True)


plt.savefig(
    "salidzinajums.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
