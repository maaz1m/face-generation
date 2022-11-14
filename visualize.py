import tensorflow as tf
import matplotlib.pyplot as plt

dataset = tf.keras.preprocessing.image_dataset_from_directory('data/img_align_celeba', label_mode=None, image_size=(32,32), batch_size=10)

plt.figure(figsize=(4, 2))
for images in dataset.take(1):
  for i in range(10):
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.axis("off")
plt.tight_layout()
plt.show()

