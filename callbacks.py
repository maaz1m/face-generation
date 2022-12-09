import tensorflow as tf
import matplotlib.pyplot as plt
import time

class VisualizeImages(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim=128):
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(10, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255

        plt.figure(figsize=(4, 2))
        for i in range(10):
            ax = plt.subplot(2, 5, i + 1)
            plt.imshow(generated_images[i].numpy().astype("uint8"))
            plt.axis("off")
        plt.suptitle(f"Epoch: {epoch} Disc loss: {logs['discriminator_loss']:.2f} Gen loss: {logs['generator_loss']:.2f}")
        plt.tight_layout()
        if epoch % 10 == 0:
            plt.savefig(f'results/results-{time.strftime("%Y%m%d%H%M%S")}.png')
            self.model.generator.save(f'saved/WGAN/generator-{time.strftime("%Y%m%d%-H%M%S")}')
        plt.show()
