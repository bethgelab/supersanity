import matplotlib.pyplot as plt

# === Fonts for print ===
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

# === Data ===
dur = [10, 30, 60, 120, 240]

mem_with =  [19, 22, 23, 24, 25]
mem_without = [33, 39, 57, 140, None]   # Last point OOM
mem_siglip = [6.5, 6.9, 7.0, 7.0, 7.2]            # Constant 5GB

# === Colors matching previous plot ===
col_limit = "#999999"

# original colors
col_g25 = "#FF8C00"
col_g20 = "#245DBD"
col_c_wo = "#B46A3B"
col_c_w  = "#8A3F1E"

# new model colors
col_sig = "#3A8F3A"


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
h1, = plt.plot(dur, mem_with,"-", color=col_c_w, linewidth=3,label="Cambrian-S (w/ Mem.)")
plt.scatter(dur, mem_with, s=100, color=col_c_w, edgecolors='white', zorder=10)

h2, = plt.plot(dur[:-1], mem_without[:-1], "--", color=col_c_wo, linewidth=3,label="Cambrian-S (w/o Mem.)")
plt.scatter(dur[:-1], mem_without[:-1], s=100, color=col_c_wo, edgecolors='white', zorder=10)

# OOM marker (front)
plt.scatter([dur[-1]], [140],marker="x", s=120,color=col_c_wo, linewidths=3,zorder=10)

# OOM label BELOW the X
plt.text(dur[-1], 140 - 5, "OOM",fontsize=14, color=col_c_wo,ha="center", va="top",zorder=11)

# === SigLIP2-So400m-512 (thick + big star) ===
h3, = plt.plot(dur, mem_siglip, "-", markersize=18, color=col_sig, linewidth=3,label=r"$\mathbf{NoSense\ (Ours)}$",)
plt.scatter(dur, mem_siglip, s=600, color=col_sig, edgecolors='white', zorder=10, marker="*")

# === Memory limit line ===
plt.axhline(140, color=col_limit, linewidth=3)
plt.text(10, 145, "Memory Limitation", fontsize=15, color="#777777")

# === Labels ===
plt.xlabel("Video Duration (in Mins.)", fontsize=16)
plt.ylabel("Memory Usage (GB)", fontsize=16)
plt.ylim(0, 155)

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
    Line2D([0], [0], color=col_c_w, linewidth=2, marker='o',
           markersize=8, markerfacecolor=col_c_w, markeredgecolor='white',
           label="Cambrian-S (w/ Mem.)"),

    Line2D([0], [0], color=col_c_wo, linewidth=2, linestyle='--', marker='o',
           markersize=8, markerfacecolor=col_c_wo, markeredgecolor='white',
           label="Cambrian-S (w/o Mem.)"),

    Line2D([0], [0], color=col_sig, linewidth=2, marker='*',
           markersize=16, markerfacecolor=col_sig, markeredgecolor='white',
           label=r"$\mathbf{NoSense\ (Ours)}$",)
]

plt.legend(
    handles=legend_handles,
    bbox_to_anchor=(0.94, 0.55),
    fontsize=12,
    title="Models",
    title_fontsize=14,
    labelspacing=1
)

plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig("memory_usage.pdf", dpi=300, transparent=True)
plt.savefig("memory_usage.png", dpi=400, transparent=True)
plt.show()
