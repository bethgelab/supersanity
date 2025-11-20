import matplotlib.pyplot as plt

# === Fonts for print ===
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

# === Data ===
dur = [0, 1, 2, 5]

perf = [42.0, 20.6, 3.6, 0]

# === Colors matching previous plot ===
col = "#8A3F1E"    # Cambrian w/ Mem

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

# === Cambrian Models (thin) ===
h1, = plt.plot(dur, perf,"-", color=col, linewidth=3,label="Cambrian-S (w/ Surprise Seg)")
plt.scatter(dur, perf, s=100, color=col, edgecolors='white', zorder=10)

# === Labels ===
plt.xlabel("Number of repeats", fontsize=16)
plt.ylabel("MRA", fontsize=16)
plt.ylim(-1.2, 45)

# === Legend (same style as accuracy plot) ===
from matplotlib.lines import Line2D
# handles = [h1, h2, h3]
# labels = ["Cambrian-S (w/ Mem.)",
#           "Cambrian-S (w/o Mem.)",
#           "NoSense"]

# plt.legend(handles, labels,
#         #    loc="center left",
#            bbox_to_anchor=(0.94, 0.55),
#            fontsize=12,
#            title="Models",
#            title_fontsize=14,
#            labelspacing=1)

legend_handles = [
    Line2D([0], [0], color=col, linewidth=2, marker='o',
           markersize=8, markerfacecolor=col, markeredgecolor='white',
           label="Cambrian-S (w/ Mem.)"),

    # Line2D([0], [0], color=col_c_wo, linewidth=2, linestyle='--', marker='o',
    #        markersize=8, markerfacecolor=col_c_wo, markeredgecolor='white',
    #        label="Cambrian-S (w/o Mem.)"),

    # Line2D([0], [0], color=col_sig, linewidth=2, marker='*',
    #        markersize=16, markerfacecolor=col_sig, markeredgecolor='white',
    #        label=r"$\mathbf{NoSense\ (Ours)}$",)
]

# plt.legend(
#     handles=legend_handles,
#     bbox_to_anchor=(0.94, 0.55),
#     fontsize=12,
#     title="Model",
#     title_fontsize=14,
#     labelspacing=1
# )

plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("vsc_repeat_mra.pdf", dpi=300, transparent=True)
plt.savefig("vsc_repeat_mra.png", dpi=400, transparent=True)
plt.show()
