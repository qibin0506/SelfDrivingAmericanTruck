import sys, getopt, time
import tensorflow as tf
from model import create_model
from predict import Predict
import utils

model = tf.keras.models.load_model(utils.model_path)

opts, _ = getopt.getopt(sys.argv[1:], '', ['dir=', 'region='])

region = utils.default_region
image_box = utils.default_image_box
map_box = utils.default_map_box

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
