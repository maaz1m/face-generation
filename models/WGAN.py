import tensorflow as tf


class WGAN(tf.keras.Model):

    def __init__(self, latent_dim):
        super(WGAN, self).__init__()
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
        super(WGAN, self).compile()
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    def gradient_penalty(self, batch_size, real_images, fake_images):

        # Get the interpolated image
        alpha = tf.random.normal(shape=[batch_size, 1, 1, 1], mean=0.0, stddev=1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3])+1e-12)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def discriminator_loss_fn(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss


    # Define the loss functions for the generator.
    def generator_loss_fn(self, fake_img):
        return -tf.reduce_mean(fake_img)

    def train_step(self, real_images):

        batch_size = tf.shape(real_images)[0]

        # train discriminator

        for i in range(3):
            with tf.GradientTape() as tape:
                random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
                fake_images = self.generator(random_latent_vectors, training=True)
                real_logits = self.discriminator(real_images, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                discriminator_loss = self.discriminator_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                discriminator_loss = discriminator_loss + gp * 10

            discriminator_gradients = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        # train generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_logits = self.discriminator(fake_images, training=True)
            generator_loss = self.generator_loss_fn(fake_logits)
        generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        return {'discriminator_loss': discriminator_loss, 'generator_loss': generator_loss}
