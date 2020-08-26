from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import parse_load


# paths
target_id = 10253
raw_fp = f"../data/raw_data/{target_id}"

# load data
raw_fp = Path(raw_fp)
record = []
arr_list = []
for fp in raw_fp.glob("**/*.png"):
    d, arr = parse_load(fp)
    record.append(d)
    arr_list.append(arr)

df = pd.DataFrame.from_records(record)

# compose
patch_shape = arr_list[0].shape
patch_dtype = arr_list[0].dtype
wsi_shape = (
    df["y"].max() - 1 + patch_shape[1],
    df["x"].max() - 1 + patch_shape[0],
    patch_shape[2],
)
wsi = np.zeros(wsi_shape, dtype=patch_dtype)
wsi_label = np.zeros(wsi_shape[:-1], dtype=int)

for i, row in df.iterrows():
    xl, yl = row["y"] - 1, row["x"] - 1
    xu, yu = xl + patch_shape[0], yl + patch_shape[1]
    wsi[xl:xu, yl:yu, :] = arr_list[i][:]
    wsi_label[xl:xu, yl:yu] = row["label"] + 1  # zero reserved for no content area

# save to disk
plt.figure()
plt.imshow(wsi)
plt.axis("off")
plt.savefig("../figures/wsi.png")
plt.close()

plt.figure()
plt.imshow(wsi_label, cmap="tab10")
plt.axis("off")
plt.savefig("../figures/wsi_label.png")
plt.close()
