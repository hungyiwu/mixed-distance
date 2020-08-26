import json
import itertools

import numpy as np
import tensorflow.keras as tfk

from skimage import img_as_float32
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from autoencoder import conv_ae


# paths
data_fp = "../data/derived_data/img_stack.npy"
checkpoint_fp = "../data/checkpoint"

# parameters
# data
test_size = 0.1
# model
latent_dim = 50
num_conv_layer = 3
num_conv_filter = 64
log_lr = -5
# train
epochs = 200
batch_size = 128
validation_split = 0.3

# load data & resize
arr = np.load(data_fp)
arr = img_as_float32(arr)
arr = resize(arr, (arr.shape[0], 64, 64, arr.shape[3]))

# manual data augmentation
x_list = []
bool_pair = [True, False]

for rot, flip0, flip1 in itertools.product(bool_pair, bool_pair, bool_pair, repeat=1):
    x = arr.copy()
    if rot:
        x = np.rot90(x, k=1, axes=(1, 2))
    if flip0:
        x = np.flip(x, axis=1)
    if flip1:
        x = np.flip(x, axis=2)
    x_list.append(x)

x = np.concatenate(x_list, axis=0)

# shuffle and split
np.random.shuffle(x)
x_train, x_test = train_test_split(x, test_size=test_size)

# load model
input_shape = x.shape[1:]
model = conv_ae(
    input_shape=input_shape,
    latent_dim=latent_dim,
    num_conv_layer=num_conv_layer,
    num_conv_filter=num_conv_filter,
)
model.compile(optimizer=tfk.optimizers.Adam(learning_rate=10 ** log_lr))

model.encoder.summary()

model.decoder.summary()

# train
print("pre-train eval")
model.evaluate(
    x=x_test, y=x_test, verbose=2, batch_size=batch_size,
)

print("train...")
history = model.fit(
    x=x_train,
    y=x_train,
    epochs=epochs,
    validation_split=validation_split,
    verbose=0,
    batch_size=batch_size,
)

print("post-train eval")
model.evaluate(
    x=x_test, y=x_test, verbose=2, batch_size=batch_size,
)

model.save_weights(checkpoint_fp / "model")

config = dict(
    num_conv_layer=num_conv_layer,
    num_conv_filter=num_conv_filter,
    input_shape=input_shape,
    latent_dim=latent_dim,
)
with open(checkpoint_fp / "config.json", "w") as f:
    json.dump(config, f)
