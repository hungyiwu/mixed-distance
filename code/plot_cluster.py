import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# load data
pc_c = np.load("../derived_data/pc_frac0.0.npy")
pc_cs = np.load("../derived_data/pc_frac0.6.npy")
label = np.load("../derived_data/label.npy")

# plot
cmap = plt.get_cmap("coolwarm")
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(9.6, 4.8))
fig.suptitle("Addition of spatial info reflected in UMAP")
common_param = dict(c=label, cmap=cmap, s=5)

axes[0].scatter(pc_c[:, 0], pc_c[:, 1], **common_param)
axes[0].set_title("0% spatial info (only content)")

axes[1].scatter(pc_cs[:, 0], pc_cs[:, 1], **common_param)
axes[1].set_title("60% spatial info")

# format axes
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

# label for brevity
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")

# label is binary, but plt.cmap takes uint8
label_negative, label_positive = sorted(np.unique(label))
legend_element = [
    Line2D(
        [0],
        [0],
        color="white",
        marker="o",
        markerfacecolor=cmap(label_positive * 255),
        label="positive",
    ),
    Line2D(
        [0],
        [0],
        color="white",
        marker="o",
        markerfacecolor=cmap(label_negative * 255),
        label="negative",
    ),
]
axes[0].legend(handles=legend_element)

fig.tight_layout()
plt.savefig("../figures/plot_cluster.png")
