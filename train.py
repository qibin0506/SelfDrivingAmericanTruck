import tensorflow as tf

from data_loader import get_batch_fn
from model import create_model
import utils

model = create_model(True)
optimizer = tf.keras.optimizers.Adam(utils.lr)
loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss_metric = tf.keras.metrics.Mean("loss_metric")
accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy_metric")


ckpt_path = "./ckpt/train/"
ckpt = tf.train.Checkpoint(
    model=model,
    optimizer=optimizer
)

ckpt_manger = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=None)
if ckpt_manger.latest_checkpoint:
    ckpt.restore(ckpt_manger.latest_checkpoint)
    print('Latest checkpoint restored')


@tf.function
def train_step(images, maps, keys):
    with tf.GradientTape() as tape:
        pred = model([images, maps])
        losses = loss_obj(y_true=keys, y_pred=pred)

    grads = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_metric(losses)
    accuracy_metric(keys, pred)


batch_fn, batch_count = get_batch_fn(utils.batch_size)
for epoch in range(utils.epochs):
    cur_batch = 0
    for images, maps, keys in batch_fn():
        train_step(images, maps, keys)
        print("epoch: {}, batch: {}/{}, loss: {}, accuracy: {}".
              format(epoch, cur_batch, batch_count, loss_metric.result(), accuracy_metric.result()))
        cur_batch += 1

    save_path = ckpt_manger.save()
