import sys, getopt, time
import tensorflow as tf
from model import create_model
from predict import Predict
import utils

model = tf.keras.models.load_model(utils.model_path)

opts, _ = getopt.getopt(sys.argv[1:], '', ['region=', 'image_box=', 'map_box=', 'use_map='])

region = utils.default_region
image_box = utils.default_image_box
map_box = utils.default_map_box
use_map = True

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
    elif opt == '--use_map':
        use_map = int(arg) == 1

print("use navigation map? {}".format(use_map))

count_done = 10
for i in range(count_done):
    print(i)
    time.sleep(1)

Predict(region=region, image_box=image_box, map_box=map_box, model=model, use_map=use_map).start()
