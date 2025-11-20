import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Fonts for print ===
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

# Path to your CSV
csv_path = "counts.csv"

# original colors
col_g25 = "#FF8C00"
col_g20 = "#245DBD"
col_c_wo = "#B46A3B"
col_c_w  = "#8A3F1E"

# new model colors
col_sig = "#3A8F3A"

# Load the data
df = pd.read_csv(csv_path)

# Drop any rows where original prediction is 0 to avoid division by zero
df = df[df["prediction_original"] != 0].copy()

# Compute values relative to the original prediction for each doc_id
df["rel_original"] = 1.0  # by definition
df["rel_answer"] = df["ground_truth"] / df["prediction_original"]
df["rel_repeat1"] = df["prediction_repeat1"] / df["prediction_original"]
df["rel_repeat2"] = df["prediction_repeat2"] / df["prediction_original"]
df["rel_repeat5"] = df["prediction_repeat5"] / df["prediction_original"]

# (Optional) inspect the first few normalized rows
print(df[["doc_id", "rel_original", "rel_answer", "rel_repeat1", "rel_repeat2", "rel_repeat5"]].head())

# ------------------------------------------------------
# 1) Per-doc_id plot of relative values (original = 1x)
# ------------------------------------------------------
plt.figure(figsize=(6,6))
ax = plt.gca()

# Remove right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# === Tick font sizes ===
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

x = df["doc_id"]

plt.plot(x, df["rel_original"], marker="o", label="Original (1x)")
plt.plot(x, df["rel_answer"], marker="o", label="Answer (GT)")
plt.plot(x, df["rel_repeat1"], marker="o", label="2x (Repeat once)")
plt.plot(x, df["rel_repeat2"], marker="o", label="3x (Repeat twice)")
plt.plot(x, df["rel_repeat5"], marker="o", label="6x (Repeat five times)")

plt.xlabel("doc_id")
plt.ylabel("Relative value (vs original prediction)")
plt.title("Per-sample values relative to original prediction")
plt.legend()
plt.tight_layout()

# ------------------------------------------------------
# 2) Average across samples, move Answer left of Original,
#    and add a linear fit through Original + repeats
# ------------------------------------------------------

# Compute means of the relative values
means = {
    "answer": df["rel_answer"].mean(),
    "original": df["rel_original"].mean(),
    "repeat1": df["rel_repeat1"].mean(),
    "repeat2": df["rel_repeat2"].mean(),
    "repeat5": df["rel_repeat5"].mean(),
}

# Choose x-positions so that Answer is to the left of Original
x_pos = {
    "answer": -1,   # left of original
    "original": 0,
    "repeat1": 1,
    "repeat2": 2,
    "repeat5": 5,   # keep 5 to reflect "repeat5" if you like
}

# Data for linear fit: ONLY original + repeats
x_fit = np.array([
    x_pos["original"],
    x_pos["repeat1"],
    x_pos["repeat2"],
    x_pos["repeat5"],
], dtype=float)

y_fit = np.array([
    means["original"],
    means["repeat1"],
    means["repeat2"],
    means["repeat5"],
], dtype=float)

# Linear regression (degree 1 polynomial)
coeffs = np.polyfit(x_fit, y_fit, 1)   # slope and intercept
poly_fn = np.poly1d(coeffs)

# For drawing the fitted line
x_line = np.linspace(min(x_fit) - 0.5, max(x_fit) + 0.5, 200)
y_line = poly_fn(x_line)

plt.figure(figsize=(6, 6))
ax = plt.gca()

# Remove right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# === Tick font sizes ===
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Scatter points for all conditions
plt.scatter([x_pos["answer"]], [means["answer"]], marker="*", s=480,  label="Answer (GT)", color=col_c_w, edgecolors='white', zorder=10)
plt.scatter(x_fit, y_fit, s=150, marker="o", label="Original + repeats", color=col_sig, edgecolors='white', zorder=10)

# Connect original/repeats with a line for visibility
#plt.plot(x_fit, y_fit)

# Plot the linear fit through original + repeats
plt.plot(x_line, y_line, linestyle="--", linewidth=2, alpha=0.6, label="Linear fit (orig + repeats)", color=col_sig)

# X-axis ticks and labels in desired order
xticks = [
    x_pos["answer"],
    x_pos["original"],
    x_pos["repeat1"],
    x_pos["repeat2"],
    x_pos["repeat5"],
]
xticklabels = ["GT", "Orig", "2x", "3x ", "6x"]
plt.xticks(xticks, xticklabels)

# Baseline at 1x
# plt.axhline(1.0, linestyle=":", linewidth=3, color=col_c_w, alpha=0.6)

plt.grid(True, linestyle="--", alpha=0.35)

plt.ylabel("Average relative value (vs original prediction)", fontsize=16)
# plt.title("Repeating k times (k+1 x long) linearly increases predicted objects")
plt.legend(fontsize=12, labelspacing=1)
plt.tight_layout()

plt.savefig("average_relative_predictions.png", dpi=300, bbox_inches='tight', transparent=True)
# plt.savefig("average_relative_predictions.png", dpi=300, bbox_inches='tight')
