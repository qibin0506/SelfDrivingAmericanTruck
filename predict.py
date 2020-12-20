import numpy as np
import cv2

from keys import get_key, press
from cap_screen import capture
import utils


class Predict(object):

    def __init__(self,
                 region,
                 image_box,
                 map_box,
                 model):

        self.region = region
        self.image_box = image_box
        self.map_box = map_box
        self.model = model

    def start(self):
        img_seq = []
        while True:
            img, _ = capture(region=self.region, dump=False)
            image, map = self.__split_and_save(img)

            image = self.__process_image(image)
            map = self.__process_map(map)

            if len(img_seq) < utils.image_seq_size:
                img_seq.append(image)
                continue

            if len(img_seq) > utils.image_seq_size:
                img_seq.pop(0)

            img_seq_input = np.array([img_seq])
            map_input = np.array([map])

            # [(batch, seq, img_height, img_width, channel) (batch, map_height, map_width, channel)]
            pred = self.model.predict([img_seq_input, map_input])[0]
            max_index = np.argmax(pred)
            print(max_index)

            key = get_key(max_index)
            press(key)

    def __split_and_save(self, img):
        image = img.crop(self.image_box)
        map = img.crop(self.map_box)

        return image, map

    def __process_image(self, img):
        img = np.asarray(img)
        img = cv2.resize(img, (utils.image_width, utils.image_height))
        img = img / 127.5 - 1.0

        return img

    def __process_map(self, img):
        img = np.asarray(img)
        img = cv2.resize(img, (utils.map_width, utils.map_height))
        img = img / 127.5 - 1.0

        return img
