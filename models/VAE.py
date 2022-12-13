<<<<<<< Updated upstream

####################################################################
#
# ADAPTED FROM Hands-On Image Generation with TensorFlow, Chapter 2
#
####################################################################

import tensorflow as tf
from tensorflow.keras import layers, Model

# Probability Distribution
class GaussianSampling(tf.keras.layers.Layer):
    def call(self, inputs):
        means, var = inputs
        eps = tf.random.normal(shape=tf.shape(means), mean=0.0, stddev=1.0)
        samples = means + tf.exp(0.5 * var) * eps
        return samples

# Down-sampling
class DownConvBlock(tf.keras.layers.Layer):
=======
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, Input, Conv2D, Dense, Flatten, Reshape, Lambda, Dropout
from tensorflow.keras.layers import Conv2DTranspose, MaxPooling2D, UpSampling2D, LeakyReLU, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class GaussianSampling(Layer):
    def call(self, inputs):
        means, logvar = inputs
        epsilon = tf.random.normal(shape=tf.shape(means), mean=0., stddev=1.)
        samples = means + tf.exp(0.5 * logvar) * epsilon

        return samples


class DownConvBlock(Layer):
>>>>>>> Stashed changes
    count = 0

    def __init__(self, filters, kernel_size=(3, 3), strides=1, padding='same'):
        super(DownConvBlock, self).__init__(name=f"DownConvBlock_{DownConvBlock.count}")
        DownConvBlock.count += 1
<<<<<<< Updated upstream
        self.forward = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)])
        self.forward.add(tf.keras.layers.BatchNormalization())
        self.forward.add(layers.LeakyReLU(0.2))
    def call(self, inputs):
        return self.forward(inputs)

# Up-sampling
class UpConvBlock(tf.keras.layers.Layer):
=======
        self.forward = Sequential([Conv2D(filters, kernel_size, strides, padding)])
        self.forward.add(BatchNormalization())
        self.forward.add(layers.LeakyReLU(0.2))

    def call(self, inputs):
        return self.forward(inputs)


class UpConvBlock(Layer):
>>>>>>> Stashed changes
    count = 0

    def __init__(self, filters, kernel_size=(3, 3), padding='same'):
        super(UpConvBlock, self).__init__(name=f"UpConvBlock_{UpConvBlock.count}")
        UpConvBlock.count += 1
<<<<<<< Updated upstream
        self.forward = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters, kernel_size, 1, padding),])
        self.forward.add(layers.LeakyReLU(0.2))
        self.forward.add(tf.keras.layers.UpSampling2D((2, 2)))
    def call(self, inputs):
        return self.forward(inputs)

# Building the Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, z_dim, name='encoder'):
        super(Encoder, self).__init__(name=name)

        # 4 Conv layers
        self.features_extract = tf.keras.models.Sequential([
=======
        self.forward = Sequential([Conv2D(filters, kernel_size, 1, padding), ])
        self.forward.add(layers.LeakyReLU(0.2))
        self.forward.add(UpSampling2D((2, 2)))

    def call(self, inputs):
        return self.forward(inputs)


class Encoder(Layer):
    def __init__(self, z_dim, name='encoder'):
        super(Encoder, self).__init__(name=name)

        self.features_extract = Sequential([
>>>>>>> Stashed changes
            DownConvBlock(filters=32, kernel_size=(3, 3), strides=2),
            DownConvBlock(filters=32, kernel_size=(3, 3), strides=2),
            DownConvBlock(filters=64, kernel_size=(3, 3), strides=2),
            DownConvBlock(filters=64, kernel_size=(3, 3), strides=2),
<<<<<<< Updated upstream
            tf.keras.layers.Flatten()])
        self.dense_mean = tf.keras.layers.Dense(z_dim, name='mean')
        self.dense_var = tf.keras.layers.Dense(z_dim, name='var')
        self.sampler = GaussianSampling()
    def call(self, inputs):
        x = self.features_extract(inputs)
        mean = self.dense_mean(x)
        var = self.dense_var(x)
        z = self.sampler([mean, var])
        return z, mean, var

# Building the Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, z_dim, name='decoder'):
        super(Decoder, self).__init__(name=name)

        self.forward = tf.keras.models.Sequential([
            tf.keras.layers.Dense(7 * 7 * 64, activation='relu'),
            tf.keras.layers.Reshape((7, 7, 64)),
            # 4 Conv layers
=======
            Flatten()])

        self.dense_mean = Dense(z_dim, name='mean')
        self.dense_logvar = Dense(z_dim, name='logvar')
        self.sampler = GaussianSampling()

    def call(self, inputs):
        x = self.features_extract(inputs)
        mean = self.dense_mean(x)
        logvar = self.dense_logvar(x)
        z = self.sampler([mean, logvar])
        return z, mean, logvar


class Decoder(Layer):
    def __init__(self, z_dim, name='decoder'):
        super(Decoder, self).__init__(name=name)

        self.forward = Sequential([
            Dense(7 * 7 * 64, activation='relu'),
            Reshape((7, 7, 64)),
>>>>>>> Stashed changes
            UpConvBlock(filters=64, kernel_size=(3, 3)),
            UpConvBlock(filters=64, kernel_size=(3, 3)),
            UpConvBlock(filters=32, kernel_size=(3, 3)),
            UpConvBlock(filters=32, kernel_size=(3, 3)),
<<<<<<< Updated upstream
            tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding='same', activation='sigmoid'),])
    def call(self, inputs):
        return self.forward(inputs)

=======
            Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding='same', activation='sigmoid'),

        ])

    def call(self, inputs):
        return self.forward(inputs)


>>>>>>> Stashed changes
class VAE(Model):
    def __init__(self, z_dim, name='VAE'):
        super(VAE, self).__init__(name=name)
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.mean = None
<<<<<<< Updated upstream
        self.var = None

    def call(self, inputs):
        z, self.mean, self.var = self.encoder(inputs)
=======
        self.logvar = None

    def call(self, inputs):
        z, self.mean, self.logvar = self.encoder(inputs)
>>>>>>> Stashed changes
        out = self.decoder(z)
        return out

