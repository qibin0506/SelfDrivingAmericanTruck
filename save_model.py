import tensorflow as tf
from model import create_model
import utils

model = create_model(True)

ckpt = tf.train.Checkpoint(model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, utils.ckpt_path, 5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).assert_consumed()
    print("Last checkpoint restored {}".format(ckpt_manager.latest_checkpoint))

    model.save(utils.model_path)
    print("model save success")