import matplotlib.pyplot as plt

# Use serif fonts for print
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

# Data
dur = [10, 30, 60, 120, 240]

# original colors
col_g25 = "#FF8C00"
col_g20 = "#245DBD"
col_c_wo = "#B46A3B"
col_c_w  = "#8A3F1E"

# new model colors
col_clip = "#4D3838"
col_b16  = "#AB1CCF"
col_so384 = "#C70039"
col_so512 = "#3A8F3A"

plt.figure(figsize=(6,6))

ax = plt.gca()

# Remove right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# Axis tick fontsize
plt.xticks(dur, fontsize=14)
plt.yticks(fontsize=14)

# Original models
h1, = plt.plot(dur,[90,82,42,0,0],"--",color=col_g25,linewidth=3,label="Gemini-2.5-Flash")
plt.scatter(dur,[90,82,42,0,0], s=100, color=col_g25, edgecolors='white', zorder=10)

h2, = plt.plot(dur,[43,40,28,22,20],"--",color=col_g20,linewidth=3,label="Gemini-2.0-Flash")
plt.scatter(dur,[43,40,28,22,20], s=100, color=col_g20, edgecolors='white', zorder=10)

h3, = plt.plot(dur,[32,28,0,0,0],"--",color=col_c_wo,linewidth=3,label="Cambrian-S (w/o Mem.)")
plt.scatter(dur,[32,28,0,0,0], s=100, color=col_c_wo, edgecolors='white', zorder=10)

h4, = plt.plot(dur,[45,42,40,40,40],"-",color=col_c_w,linewidth=3,label="Cambrian-S (w/ Mem.)")
plt.scatter(dur,[45,42,40,40,40], s=100, color=col_c_w, edgecolors='white', zorder=10)

# New models
# h5, = plt.plot(dur,[91.67,83.33,80,76.67,71.67],"o:",color=col_clip,linewidth=5,label="CLIP-L/14-224")
# h6, = plt.plot(dur,[95,91.67,88.33,83.33,85],"o:",color=col_b16,linewidth=5,label="SigLIP2-B/16-224")
# h7, = plt.plot(dur,[98.33,98.33,95,95,90],"o:",color=col_so384,linewidth=5,label="SigLIP2-So400m-384")

# Best model
h8, = plt.plot(dur,[98.33,98.33,96.67,95,95], "-", markersize=18,color=col_so512,linewidth=3,label=r"$\mathbf{NoSense\ (Ours)}$",)
plt.scatter(dur,[98.33,98.33,96.67,95,95], s=600, color=col_so512, edgecolors='white', zorder=10, marker="*")


plt.xlabel("Video Duration (in Mins.)", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.ylim(-1.2,105)
plt.grid(True, linestyle="--", alpha=0.35)

# Legend grouping with gap
original_handles = [h1, h2, h3, h4]
# new_handles = [h5, h6, h7, h8]
new_handles = [h8]

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0],
           color=col_g25, linewidth=2, linestyle="--",
           marker='o', markersize=8,
           markerfacecolor=col_g25, markeredgecolor='white',
           label="Gemini-2.5-Flash"),

    Line2D([0], [0],
           color=col_g20, linewidth=2, linestyle="--",
           marker='o', markersize=8,
           markerfacecolor=col_g20, markeredgecolor='white',
           label="Gemini-2.0-Flash"),

    Line2D([0], [0],
           color=col_c_wo, linewidth=2, linestyle="--",
           marker='o', markersize=8,
           markerfacecolor=col_c_wo, markeredgecolor='white',
           label="Cambrian-S (w/o Mem.)"),

    Line2D([0], [0],
           color=col_c_w, linewidth=2, linestyle="-",
           marker='o', markersize=8,
           markerfacecolor=col_c_w, markeredgecolor='white',
           label="Cambrian-S (w/ Mem.)"),

    Line2D([0], [0],
           color=col_so512, linewidth=2, linestyle='-',
           marker='*', markersize=16,
           markerfacecolor=col_so512, markeredgecolor='white',
           label=r"$\mathbf{NoSense\ (Ours)}$",),
]

plt.legend(
    handles=legend_handles,
    bbox_to_anchor=(0.35, 0.45),
    fontsize=12,
    title="Models",
    title_fontsize=13,
    labelspacing=0.75,
    handletextpad=1,
    borderpad=0.5
)

plt.tight_layout()
plt.savefig("accuracy.pdf", dpi=300, transparent=True)
plt.savefig("accuracy.png", dpi=400, transparent=True)
plt.show()
