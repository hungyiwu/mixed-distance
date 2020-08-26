from pathlib import Path

import numpy as np
import pandas as pd

from umap import UMAP
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# paths
data_fp = "../data/derived_data"

# load data
data_fp = Path(data_fp)
z = np.load(data_fp / "img_stack_latent.npy")

metadata = pd.read_csv(data_fp / "metadata.csv")

# filter for single WSI
mask = metadata["filename"].apply(lambda x: x.startswith("10253_"))
z = z[mask, ...]
coord = metadata.loc[mask, ["x", "y"]].values
label = metadata.loc[mask, "label"].values

# normalize coord to same scale of latent vectors
# for balanced UMAP later
target_mean, target_std = z.mean(), z.std()
coord = coord.astype(z.dtype)
coord -= coord.mean(axis=0) - target_mean
coord /= coord.std(axis=0) / target_std

# umap sweep fraction
frac_list = np.linspace(0, 1, num=50)
record = []
cv_fold = 5
metric_list = ["accuracy", "precision", "recall"]
for frac in frac_list:
    model = UMAP(n_components=2, target_metric="l2", target_weight=frac)
    pc = model.fit_transform(z, y=coord)
    res = [frac]
    for metric in metric_list:
        values = cross_val_score(estimator=SVC(), X=pc, y=label, cv=cv_fold,
                                 scoring=metric)
        res.extend([values.mean(), values.std()])
    record.append(res)

# save to disk
col_list = ["frac_spatial"]
for metric in metric_list:
    col_list.extend([f"{metric}_mean", f"{metric}_std"])
df = pd.DataFrame.from_records(record, columns=col_list)
df.to_csv("../derived_data/result.csv", index=False)

# save pc to disk
for frac in [0, 0.6]:
    model = UMAP(n_components=2, target_metric="l2", target_weight=frac)
    pc = model.fit_transform(z, y=coord)
    np.save(f"../derived_data/pc_frac{frac:.1f}.npy", pc)
np.save("../derived_data/label.npy", label)
