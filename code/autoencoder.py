import tensorflow as tf
import tensorflow.keras as tfk


class conv_ae(tfk.Model):
    def __init__(self, input_shape: tuple, latent_dim: int, **kwargs):
        super(conv_ae, self).__init__(**kwargs)
        conv_1_params = dict(
            filters=32, kernel_size=3, activation="relu", padding="same"
        )
        conv_2_params = dict(
            filters=64, kernel_size=3, activation="relu", padding="same"
        )

        # encoder
        encoder_input = tfk.Input(shape=input_shape)
        x = tfk.layers.Conv2D(**conv_1_params)(encoder_input)
        x = tfk.layers.Conv2D(**conv_2_params)(x)
        intermediate_shape = x.shape[1:]
        x = tfk.layers.Flatten()(x)
        encoder_output = tfk.layers.Dense(units=latent_dim)(x)
        self.encoder = tfk.Model(encoder_input, encoder_output, name="encoder")

        # decoder
        decoder_input = tfk.Input(shape=(latent_dim,))
        x = tfk.layers.Dense(
            units=tf.math.reduce_prod(intermediate_shape), activation="relu"
        )(decoder_input)
        x = tfk.layers.Reshape(intermediate_shape)(x)
        x = tfk.layers.Conv2DTranspose(**conv_2_params)(x)
        x = tfk.layers.Conv2DTranspose(**conv_1_params)(x)
        decoder_output = tfk.layers.Conv2DTranspose(
            filters=input_shape[2], kernel_size=3, activation="sigmoid", padding="same"
        )(x)
        self.decoder = tfk.Model(decoder_input, decoder_output, name="decoder")

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.decoder(self.encoder(x))
            reconstruction_loss = tf.reduce_mean(
                tfk.losses.binary_crossentropy(y, y_pred)
            )
        gradients = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return {"loss": reconstruction_loss}

    def test_step(self, data):
        x, y = data
        y_pred = self.decoder(self.encoder(x))
        reconstruction_loss = tf.reduce_mean(tfk.losses.binary_crossentropy(y, y_pred))
        return {"loss": reconstruction_loss}

    def call(self, x):
        return self.decoder(self.encoder(x))
