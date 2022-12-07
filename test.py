from tensorflow.keras.models import load_model
from tensorflow.random import normal
import matplotlib.pyplot as plt

LATENT_DIM = 128

# load saved model
model = load_model('saved/GAN/generator')

# generate images from the generator of the loaded model
random_latent_vectors = normal(shape=(10, LATENT_DIM))
generated_images = model(random_latent_vectors)
generated_images *= 255

# visualize the images
plt.figure(figsize=(8, 4))
for i in range(10):
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(generated_images[i].numpy().astype("uint8"))
    plt.axis("off")
plt.tight_layout()
plt.show()