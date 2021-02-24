import tensorflow as tf

from data_loader import get_batch_fn
from model import create_model
import utils

model = create_model(True)

optimizer = tf.keras.optimizers.Adam(utils.get_lr(0))
loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

loss_metric = tf.keras.metrics.Mean("loss_metric")
accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy_metric")
epoch_loss_metric = tf.keras.metrics.Mean("epoch_loss_metric")
epoch_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="epoch_accuracy_metric")

ckpt = tf.train.Checkpoint(model=model)
ckpt_manger = tf.train.CheckpointManager(ckpt, utils.ckpt_path, max_to_keep=5)

if ckpt_manger.latest_checkpoint:
    ckpt.restore(ckpt_manger.latest_checkpoint)
    print('Latest checkpoint restored: {}'.format(ckpt_manger.latest_checkpoint))


@tf.function(experimental_relax_shapes=True)
def train_step(images, maps, keys):
    with tf.GradientTape() as tape:
        pred = model([images, maps], training=True)
        losses = loss_obj(y_true=keys, y_pred=pred)

    grads = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_metric.update_state(losses)
    accuracy_metric.update_state(y_true=keys, y_pred=pred)

    epoch_loss_metric.update_state(losses)
    epoch_accuracy_metric.update_state(y_true=keys, y_pred=pred)


def save_ckpt(epoch, cur_batch, batch_count, batch_size):
    loss_rst = loss_metric.result()
    accuracy_rst = accuracy_metric.result()

    loss_metric.reset_states()
    accuracy_metric.reset_states()

    print("\nepoch: {}, batch: {}/{}, batch_size: {}, lr: {}, loss: {}, accuracy: {}".
          format(epoch, cur_batch, batch_count, batch_size,
                 optimizer.lr.numpy(), loss_rst, accuracy_rst))
    save_path = ckpt_manger.save()


def train():
    on_epoch = get_batch_fn(utils.batch_size)

    for epoch in range(utils.epochs):
        cur_batch = 0
        optimizer.lr = utils.get_lr(epoch)

        batch_fn, batch_count = on_epoch(epoch)
        batch_size = 0

        for images, maps, keys in batch_fn():
            batch_size = images.shape[0]
            train_step(images, maps, keys)

            if cur_batch % 50 == 0:
                save_ckpt(epoch, cur_batch, batch_count, batch_size)
            else:
                print("\u001b[2K",
                      "\u001b[100D",
                      "epoch: {}, batch: {}/{}, batch_size: {}, lr: {}".
                      format(epoch, cur_batch, batch_count, images.shape[0], optimizer.lr.numpy()),
                      end='', flush=True)

            cur_batch += 1

        save_ckpt(epoch, cur_batch, batch_count, batch_size)

        print("summary-> epoch: {}, loss: {}, accuracy: {}"
              .format(epoch, epoch_loss_metric.result(), epoch_accuracy_metric.result()))
        epoch_loss_metric.reset_states()
        epoch_accuracy_metric.reset_states()


if __name__ == '__main__':
    train()
