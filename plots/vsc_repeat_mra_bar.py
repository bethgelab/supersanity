import matplotlib.pyplot as plt
import numpy as np

# === Fonts for print ===
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

# === Data ===
dur = [0, 1, 2, 5]

perf = [42.0, 20.6, 3.6, 0]

# === Colors matching previous plot ===
col = "#8A3F1E"    # Cambrian w/ Mem

x = np.arange(len(dur))
width = 0.35

plt.figure(figsize=(6,6))
ax = plt.gca()

# Remove right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# === Tick font sizes ===
plt.xticks(dur, fontsize=14)
plt.yticks(fontsize=14)

bars1 = ax.bar(x, perf,   width, color=col)

plt.xlabel("Number of repeats", fontsize=16)
plt.ylabel("MRA", fontsize=16)
plt.ylim(0, 45)

ax.set_xticks(x)
ax.set_xticklabels(dur)

plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("vsc_repeat_mra_bar.pdf", dpi=300, transparent=True)
plt.savefig("vsc_repeat_mra_bar.png", dpi=400, transparent=True)
