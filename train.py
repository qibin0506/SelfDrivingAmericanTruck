import tensorflow as tf

from data_loader import get_batch_fn
from model import create_model
import utils

model = create_model(True)
optimizer = tf.keras.optimizers.Adam(utils.get_lr(0))
loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss_metric = tf.keras.metrics.Mean("loss_metric")
accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy_metric")

ckpt = tf.train.Checkpoint(
    model=model
)

ckpt_manger = tf.train.CheckpointManager(ckpt, utils.ckpt_path, max_to_keep=5)
if ckpt_manger.latest_checkpoint:
    ckpt.restore(ckpt_manger.latest_checkpoint)
    print('Latest checkpoint restored: {}'.format(ckpt_manger.latest_checkpoint))


@tf.function
def train_step(images, maps, keys):
    with tf.GradientTape() as tape:
        pred = model([images, maps])
        losses = loss_obj(y_true=keys, y_pred=pred)

    grads = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_metric(losses)
    accuracy_metric(y_true=keys, y_pred=pred)


batch_fn, batch_count = get_batch_fn(utils.batch_size)
for epoch in range(utils.epochs):
    cur_batch = 0
    optimizer.lr = utils.get_lr(epoch)

    for images, maps, keys in batch_fn():
        train_step(images, maps, keys)

        loss_rst = loss_metric.result()
        print("epoch: {}, batch: {}/{}, loss: {}, accuracy: {}, lr: {}".
              format(epoch, cur_batch, batch_count, loss_rst, accuracy_metric.result(), optimizer.lr.numpy()))

        cur_batch += 1

        # if loss_rst < 1:
        #     optimizer.lr = utils.lr2
        # elif loss_rst < 1.1:
        #     optimizer.lr = utils.lr1
        # else:
        #     optimizer.lr = utils.lr

        if cur_batch % 50 == 0:
            save_path = ckpt_manger.save()
