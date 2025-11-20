import matplotlib.pyplot as plt

# Use serif fonts for print
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

# Data from the table
times = [10, 30, 60, 120, 240]

# original colors
col_g25 = "#3A8F3A"
col_g20 = "#245DBD"
col_c_wo = "#B46A3B"
col_c_w  = "#8A3F1E"

# new model colors
col_clip = "#4D3838"
col_b16  = "#AB1CCF"
col_so384 = "#C70039"
col_so512 = "#FF8C00"

results = {
    "Clip-l14-224": [91.67, 83.33, 80.0, 76.67, 71.67],
    "Siglip2-b16-224": [95.0, 91.67, 88.33, 83.33, 85.0],
    "Siglip2-So400m-384": [98.33, 98.33, 95.0, 95.0, 90.0],
    "Siglip2-So400m-512": [98.33, 98.33, 96.67, 95.0, 95.0],
}

plt.figure(figsize=(6, 6))

ax = plt.gca()

# Remove right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# Axis tick fontsize
plt.xticks(times, fontsize=14)
plt.yticks(fontsize=14)

h4, = plt.plot(times,[45,42,40,40,40],"-",color=col_c_w,linewidth=2,label="Cambrian-S (w/ Mem.)")
plt.scatter(times,[45,42,40,40,40], s=100, color=col_c_w, edgecolors='white', zorder=10)

plt.plot(times, results["Clip-l14-224"], markersize=18, color=col_clip, linewidth=2, label="NoSense (CLIP-L/14)")
plt.scatter(times, results["Clip-l14-224"], s=600, color=col_clip, edgecolors="white", zorder=10, marker="*")

plt.plot(times, results["Siglip2-b16-224"], markersize=18, color=col_b16, linewidth=2, label="NoSense (Siglip2-B/16)")
plt.scatter(times, results["Siglip2-b16-224"], s=600, color=col_b16, edgecolors="white", zorder=10, marker="*")

plt.plot(times, results["Siglip2-So400m-384"], markersize=18, color=col_so384, linewidth=2, label="NoSense (Siglip2-So400m-384)")
plt.scatter(times, results["Siglip2-So400m-384"], s=600, color=col_so384, edgecolors="white", zorder=10, marker="*")

plt.plot(times, results["Siglip2-So400m-512"], markersize=18, color=col_so512, linewidth=2, label="NoSense (Siglip2-So400m-512)")
plt.scatter(times, results["Siglip2-So400m-512"], s=600, color=col_so512, edgecolors="white", zorder=10, marker="*")

plt.xlabel("Video Duration (in Mins.)", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=16)
# plt.ylim(-1.2,105)
plt.grid(True, linestyle="--", alpha=0.35)

plt.legend(
    bbox_to_anchor=(1.0, 0.45),
    fontsize=12,
    title_fontsize=13,
    labelspacing=0.75,
    handletextpad=1,
    borderpad=0.5
)

plt.xticks(times, [f"{t}" for t in times])
# plt.legend()
plt.tight_layout()
plt.savefig("accuracy_our_model_abl.pdf", dpi=300, bbox_inches='tight')
plt.savefig("accuracy_our_model_abl.png", dpi=400, bbox_inches='tight')
