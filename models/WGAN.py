import tensorflow as tf


class WGAN(tf.keras.Model):

    def __init__(self, latent_dim, d_steps, gp_weight):
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.d_steps = d_steps
        self.gp_weight = gp_weight

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
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.discriminator_loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.generator_loss_fn = tf.keras.losses.BinaryCrossentropy()

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp



    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_images = self.generator(random_latent_vectors)

        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        random_noise = lambda: 0.05 * tf.random.uniform((batch_size, 1))
        real_labels += random_noise()
        fake_labels += random_noise()

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                # d_cost = self.discriminator_loss_fn(real_img=real_logits, fake_img=fake_logits)
                d_cost = self.discriminator_loss_fn(real_labels, real_logits) + self.discriminator_loss_fn(
                fake_labels, fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                discriminator_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.discriminator_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            generator_loss = self.generator_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(generator_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": discriminator_loss, "g_loss": generator_loss}