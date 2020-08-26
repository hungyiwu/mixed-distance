import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_float32
from skimage.transform import resize

from autoencoder import conv_ae


# paths
checkpoint_fp = "../data/checkpoint"
data_fp = "../data/derived_data"

# params
num_img = 3

# load trained model
checkpoint_fp = Path(checkpoint_fp)
with open(checkpoint_fp / "config.json", "r") as f:
    config = json.load(f)
model = conv_ae(**config)

# model.encoder.summary()
# model.decoder.summary()

model.load_weights(checkpoint_fp / "model")

# load data & resize & reconstruct
data_fp = Path(data_fp)
arr = np.load(data_fp / "img_stack.npy")
arr = img_as_float32(arr)
arr = resize(arr, (arr.shape[0], 64, 64, arr.shape[3]))
arr_latent = model.encoder(arr)
arr_pred = model.decoder(arr_latent)

# show reconstruction result
sample = np.random.choice(range(arr.shape[0]), size=num_img, replace=False)

fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(4.8, 6.4))
imshow_param = dict(cmap="gray", vmin=0, vmax=1)

for row in range(num_img):
    i = sample[row]
    axes[row, 0].imshow(arr[i, ...], **imshow_param)
    axes[row, 1].imshow(arr_pred[i, ...], **imshow_param)

for ax in axes.reshape(-1):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

axes[0, 0].set_title("input")
axes[0, 1].set_title("output")
fig.tight_layout()
plt.savefig("reconstructed.png")

# save to disk
np.save(data_fp / "img_stack_latent.npy", arr_latent)
np.save(data_fp / "img_stack_pred.npy", arr_pred)
