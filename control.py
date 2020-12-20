import sys, getopt, time
import tensorflow as tf
from model import create_model
from predict import Predict

model = create_model(False)

ckpt_path = "./ckpt/train/"
ckpt = tf.train.Checkpoint(model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, None)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
else:
    print("checkpoint not found")
    exit(0)

opts, _ = getopt.getopt(sys.argv[1:], '', ['dir=', 'region='])

region = [318, 137, 808, 507]
image_box = [0, 268, 1606, 622]
map_box = [1218, 752, 1572, 854]

for opt, arg in opts:
    if opt == '--region':
        regions = arg.split(',')

        region = []
        for r in regions:
            region.append(int(r))
    elif opt == '--image_box':
        boxs = arg.split(',')
        image_box = []
        for b in boxs:
            image_box.append(int(b))
    elif opt == '--map_box':
        boxs = arg.split(',')
        map_box = []
        for b in boxs:
            map_box.append(int(b))

count_done = 10
for i in range(count_done):
    print(i)
    time.sleep(1)

Predict(region=region, image_box=image_box, map_box=map_box, model=model).start()
