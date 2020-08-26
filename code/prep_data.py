from pathlib import Path

import numpy as np
import pandas as pd

from util import parse_load


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
