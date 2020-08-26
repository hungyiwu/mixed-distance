from pathlib import Path
import numpy as np
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
