import json
import shutil
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import autoencoder


# paths
data_fp = "../data"

# params
validation_split = 0.3
img_shape = (50, 50, 3)  # for convenience, can get from data
latent_dim = 5
epochs = 1
batch_size = 32
optimizer = "adam"

# load data with on-the-fly augmentation
data_fp = Path(data_fp)
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255.0,
    validation_split=validation_split,
)
flow_param = dict(
    directory=data_fp / "raw_data",
    class_mode="input",
    color_mode="rgb",
    target_size=img_shape[:-1],
    shuffle=True,
    follow_links=True,
)
ds_train = datagen.flow_from_directory(subset="training", **flow_param)
ds_val = datagen.flow_from_directory(subset="validation", **flow_param)

# load model
model = autoencoder.conv_ae(input_shape=img_shape, latent_dim=latent_dim)
model.compile(optimizer=optimizer)

# train
print("pre-train eval")
model.evaluate(ds_val)

print("train")
model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=ds_val)

print("post-train eval")
model.evaluate(ds_val)

# save model and metadata
checkpoint_fp = data_fp / "checkpoint"
if checkpoint_fp.exists():
    shutil.rmtree(checkpoint_fp)

model.save_weights(checkpoint_fp / "weights")

config = dict(
    validation_split=validation_split,
    input_shape=img_shape,
    latent_dim=latent_dim,
    num_epoch=epochs,
    batch_size=batch_size,
    optimizer=optimizer,
)
with open(checkpoint_fp / "config.json", "w") as f:
    json.dump(config, f)
