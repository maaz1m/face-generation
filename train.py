import tensorflow as tf
from models.GAN import GAN
from models.WGAN import WGAN
from callbacks import VisualizeImages

model_type = 'wgan'

if model_type == 'gan':
    LATENT_DIM = 128
    NUM_EPOCHS = 30
    BATCH_SIZE = 64
    # load image files from directory as tensorflow Dataset object
    dataset = tf.keras.preprocessing.image_dataset_from_directory('data/img_align_celeba', label_mode=None, image_size=(64, 64), batch_size=BATCH_SIZE)

    # convert integer encodings to float
    dataset = dataset.map(lambda x: x / 255.)
    # create and compile model object
    model = GAN(latent_dim=LATENT_DIM)
    model.compile()

    # create callback for visualizing results
    visualize_images = VisualizeImages(latent_dim=LATENT_DIM, model_type=model_type)

    # create callback for temporary saving model after every epoch, so if training
    # is interrupted we can resume from the last completed epoch
    backup = tf.keras.callbacks.BackupAndRestore(backup_dir="backup")

    # train the model
    hist = model.fit(dataset, epochs=NUM_EPOCHS, callbacks=[visualize_images, backup])

    # save model for further training or inference
    model.generator.save('saved/GAN/generator')
elif model_type == 'wgan':
    LATENT_DIM = 128
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    # load image files from directory as tensorflow Dataset object
    dataset = tf.keras.preprocessing.image_dataset_from_directory('data/img_align_celeba', label_mode=None, image_size=(64, 64), batch_size=BATCH_SIZE)

    # convert integer encodings to float
    dataset = dataset.map(lambda x: x / 255.)
    # create and compile model object
    model = WGAN(latent_dim=LATENT_DIM)
    model.compile()

    # create callback for visualizing results
    visualize_images = VisualizeImages(latent_dim=LATENT_DIM, model_type=model_type)

    # create callback for temporary saving model after every epoch, so if training
    # is interrupted we can resume from the last completed epoch
    backup = tf.keras.callbacks.BackupAndRestore(backup_dir="backup")

    # train the model
    hist = model.fit(dataset, epochs=NUM_EPOCHS, callbacks=[visualize_images, backup])

    # save model for further training or inference
    model.generator.save('saved/WGAN/generator')
elif model_type == 'vae':
    # TODO Add VAE train code
    1
else:
    raise Exception('Model type not recognized')