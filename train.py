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
    visualize_images = VisualizeImages(latent_dim=LATENT_DIM)

    # create callback for temporary saving model after every epoch, so if training
    # is interrupted we can resume from the last completed epoch
    backup = tf.keras.callbacks.BackupAndRestore(backup_dir="backup")

    # train the model
    hist = model.fit(dataset, epochs=NUM_EPOCHS, callbacks=[visualize_images, backup])

    # save model for further training or inference
    model.generator.save('saved/GAN/generator')
    model.discriminator.save('saved/GAN/discriminator')
elif model_type == 'wgan':
    # TODO Add WGAN train code
    LATENT_DIM = 128
    D_STEPS = 3
    NUM_EPOCHS = 30
    BATCH_SIZE = 64
    GP_WEIGHT = 10.0
    # load image files from directory as tensorflow Dataset object
    dataset = tf.keras.preprocessing.image_dataset_from_directory('data/img_align_celeba', label_mode=None,
                                                                  image_size=(64, 64), batch_size=BATCH_SIZE)

    # convert integer encodings to float
    dataset = dataset.map(lambda x: x / 255.)
    # create and compile model object
    model = WGAN(latent_dim=LATENT_DIM, d_steps = D_STEPS, gp_weight= GP_WEIGHT)
    model.compile()

    # create callback for visualizing results
    visualize_images = VisualizeImages(latent_dim=LATENT_DIM)

    # create callback for temporary saving model after every epoch, so if training
    # is interrupted we can resume from the last completed epoch
    backup = tf.keras.callbacks.BackupAndRestore(backup_dir="backup")

    # train the model
    hist = model.fit(dataset, epochs=NUM_EPOCHS, callbacks=[visualize_images, backup])

    # save model for further training or inference
    model.generator.save('saved/WGAN/generator')
    model.discriminator.save('saved/WGAN/discriminator')
elif model_type == 'vae':

    ####################################################################
    #
    # ADAPTED FROM Hands-On Image Generation with TensorFlow, Chapter 2
    #
    ####################################################################

    from models.VAE import VAE
    import tensorflow_datasets as tfds

    import matplotlib.pyplot as plt
    import os
    import warnings

    warnings.filterwarnings('ignore')

    # ***********************************************************************************************************************
    #
    # NOTE: The training script will only work if you already have the 'celeb-a'
    # dataset downloaded and placed in your tensorflow_datasets folder. Downloading
    # the dataset using tf.load() is currently BUGGED and not resolved as of 12/12/2022.
    # Upgrading to tfds-nightly does NOT work.
    # To manually download the .zip file, use the following link:
    # https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=share_link
    # ***********************************************************************************************************************

    (ds_train, ds_test_), ds_info = tfds.load('celeb_a', split=['train', 'test'], shuffle_files=True, with_info=True,
                                              download=False)

    EPOCHS = 50
    BATCH_SIZE = 128
    LR = .001
    SEED = 42


    # Pre-Processing
    def preprocess(sample):
        image = sample['image']
        image = tf.image.resize(image, [112, 112])
        image = tf.cast(image, tf.float32) / 255.
        return image, image


    ds_train = ds_train.map(preprocess)
    ds_train = ds_train.shuffle(SEED)
    ds_train = ds_train.batch(BATCH_SIZE).prefetch(BATCH_SIZE)

    ds_test = ds_test_.map(preprocess).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

    len_train = ds_info.splits['train'].num_examples
    len_test = ds_info.splits['test'].num_examples

    vae = VAE(z_dim=200)


    # Calculate loss metrics
    def vae_kl_loss(y_true, y_pred):
        kl_loss = - 0.5 * tf.reduce_mean(1 + vae.var - tf.square(vae.mean) - tf.exp(vae.var))
        return kl_loss


    def vae_bce_loss(y_true, y_pred):
        bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return bce_loss


    def vae_loss(y_true, y_pred):
        kl_loss = vae_kl_loss(y_true, y_pred)
        bce_loss = vae_bce_loss(y_true, y_pred)
        kl_weight_const = 0.01
        return kl_weight_const * kl_loss + bce_loss


    model_path = "./models/Final_Project_vae_celeb_a.h5"
    os.makedirs("./models", exist_ok=True)

    # Compile and fit model

    vae.compile(loss=[vae_loss], optimizer=tf.keras.optimizers.Adadelta(learning_rate=LR),
                metrics=[vae_kl_loss, vae_bce_loss])
    history = vae.fit(ds_train, validation_data=ds_test, epochs=EPOCHS)

    images, labels = next(iter(ds_train))
    vae.load_weights(model_path)
    outputs = vae.predict(images)

    # Display 8x4 grid of train and test images
    col = 8
    row = 4

    fig, ax = plt.subplots(row, col, figsize=(col * 2, row * 2))

    i = 0
    for row in range(0, row, 2):
        for col in range(col):
            ax[row, col].imshow(images[i])
            ax[row, col].axis('off')
            ax[row + 1, col].imshow(outputs[i])
            ax[row + 1, col].axis('off')
            i += 1

    # Save result image in directory
    plt.savefig('VAE_result.png')
else:
    raise Exception('Model type not recognized')