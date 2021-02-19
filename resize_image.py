from glob import glob
import cv2
import os

import utils

images = glob("./data/images/*")
maps = glob("./data/maps/*")

for item in images:
    img = cv2.imread(item)
    img = cv2.resize(img, (utils.image_width, utils.image_height))

    os.remove(item)
    cv2.imwrite(item, img)


for item in maps:
    img = cv2.imread(item)
    img = cv2.resize(img, (utils.map_width, utils.map_height))

    os.remove(item)
    cv2.imwrite(item, img)
