import tensorflow as tf


class GAN(tf.keras.Model):

    def __init__(self, latent_dim):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

    def build_discriminator(self):
        discriminator = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(64, 64, 3)),
                tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        return discriminator

    def build_generator(self):
        generator = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.latent_dim,)),
                tf.keras.layers.Dense(8 * 8 * 128),
                tf.keras.layers.Reshape((8, 8, 128)),
                tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
            ],
            name="generator",
        )
        return generator

    def compile(self):
        super(GAN, self).compile()
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.discriminator_loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.generator_loss_fn = tf.keras.losses.BinaryCrossentropy()

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # train discriminator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_images = self.generator(random_latent_vectors)

        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        random_noise = lambda: 0.05 * tf.random.uniform((batch_size, 1))
        real_labels += random_noise()
        fake_labels += random_noise()

        with tf.GradientTape() as tape:
            real_logits = self.discriminator(real_images)
            fake_logits = self.discriminator(fake_images)
            discriminator_loss = self.discriminator_loss_fn(real_labels, real_logits) + self.discriminator_loss_fn(
                fake_labels, fake_logits)
        discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_weights))

        # train generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        real_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors)
            fake_logits = self.discriminator(fake_images)
            generator_loss = self.generator_loss_fn(real_labels, fake_logits)
        generator_gradients = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_weights))

        return {'discriminator_loss': discriminator_loss, 'generator_loss': generator_loss}
