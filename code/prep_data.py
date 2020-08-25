from pathlib import Path

import numpy as np
import pandas as pd

from skimage.io import imread


def parse_load(filepath: str) -> (dict, np.ndarray):
    """
    Parse file name and load image.

    Args:
        filepath: str
            File path.

    Return:
        dict of metadata
        image data in np.ndarray format
    """
    fp = Path(filepath)
    fields = fp.stem.split("_")
    metadata = dict(
        x=int(fields[2][1:]),
        y=int(fields[3][1:]),
        label=int(fields[4][5:]),
        filename=fp.name,
    )
    arr = imread(fp)
    return metadata, arr


# paths
data_fp = "../data"

# load data
data_fp = Path(data_fp)
img_stack = []
metadata = []

for img_fp in (data_fp / "raw_data").glob("**/*.png"):
    d, a = parse_load(img_fp)
    img_stack.append(a)
    metadata.append(d)

img_stack = np.stack(img_stack, axis=0)
metadata = pd.DataFrame.from_records(metadata)

np.save(data_fp / "derived_data" / "img_stack.npy", img_stack)
metadata.to_csv(data_fp / "derived_data" / "metadata.csv", index=False)

print(f"{img_stack.shape[0]} images parsed.")
