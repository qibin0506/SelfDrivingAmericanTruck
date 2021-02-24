import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_loader import get_image, get_map
from model import create_model
import utils


model = tf.keras.models.load_model(utils.model_path)

# model = create_model(False)
# model.summary()
#
# ckpt = tf.train.Checkpoint(model=model)
# ckpt_manger = tf.train.CheckpointManager(ckpt, utils.ckpt_path, max_to_keep=5)
#
# if ckpt_manger.latest_checkpoint:
#     ckpt.restore(ckpt_manger.latest_checkpoint)
#     print('Latest checkpoint restored: {}'.format(ckpt_manger.latest_checkpoint))


image_feature_layer = model.get_layer(name='time_distributed_1')
map_feature_layer = model.get_layer(name='batch_normalization_5')

re_model = tf.keras.models.Model(inputs=model.inputs, outputs=[image_feature_layer.input, map_feature_layer.input])


test_image_names = ['1610281340_942', '1610281341_943', '1610281342_944', '1610281343_945', '1610281344_946']
test_image_seq = []
for n in test_image_names:
    test_image_seq.append(get_image("{}".format(n)))
test_map = get_map("1610281344_946")

pred = re_model([np.array([test_image_seq]), np.array([test_map])], training=False)

image_features = pred[0].numpy()
map_features = pred[1].numpy()

for i in range(image_features.shape[1]): # seqs
    avg = np.average(image_features[0, i], axis=-1)
    plt.imshow(avg, cmap='gray')
    plt.axis('off')
    plt.show()


avg = np.average(map_features[0], axis=-1)
plt.imshow(avg, cmap='gray')
plt.axis('off')
plt.show()
