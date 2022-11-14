import tensorflow as tf
from models.GAN import GAN
from callbacks import VisualizeImages

LATENT_DIM = 128
NUM_EPOCHS = 30
BATCH_SIZE = 256

dataset = tf.keras.preprocessing.image_dataset_from_directory('data/img_align_celeba', label_mode=None, image_size=(32,32), batch_size=BATCH_SIZE)
dataset = dataset.map(lambda x: x / 255.0)

gan = GAN(latent_dim=LATENT_DIM)
gan.compile()

visualize_images = VisualizeImages(latent_dim=LATENT_DIM)
gan.fit(dataset, epochs=NUM_EPOCHS, callbacks=[visualize_images])