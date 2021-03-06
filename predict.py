import numpy as np
import cv2
import time

from keys import get_key, press
from cap_screen import capture
import utils


class Predict(object):

    def __init__(self,
                 region,
                 image_box,
                 map_box,
                 model,
                 use_map=True):

        self.region = region
        self.image_box = image_box
        self.map_box = map_box
        self.model = model
        self.pred_ret_idx = 1 if use_map else 0
        self.last_s_distance = 0

    def start(self):
        img_seq = []
        img_seq_length = (utils.image_seq_size - 1) * utils.pred_skip_frame + 1

        while True:
            img, _ = capture(region=self.region, dump=False)
            image, map = self.__split_and_save(img)

            image = self.__process_image(image)
            map = self.__process_map(map)

            img_seq.append(image)

            if len(img_seq) < img_seq_length:
                continue

            if len(img_seq) > img_seq_length:
                img_seq.pop(0)

            img_seq_input = np.array([img_seq[::utils.pred_skip_frame]])
            map_input = np.array([map])

            start_time = time.time()

            """
            Computation is done in batches. This method is designed for performance in
            large scale inputs. For small amount of inputs that fit in one batch,
            directly using `__call__` is recommended for faster execution, e.g.,
            `model(x)`, or `model(x, training=False)` if you have layers such as
            `tf.keras.layers.BatchNormalization` that behaves differently during
            inference. Also, note the fact that test loss is not affected by
            regularization layers like noise and dropout.
            """
            # pred = self.model.predict([img_seq_input, map_input])[0]

            # [(batch, seq, img_height, img_width, channel) (batch, map_height, map_width, channel)]
            pred = self.model([img_seq_input, map_input], training=False)[self.pred_ret_idx].numpy()[0]
            max_index = np.argmax(pred)
            # print(pred)
            # print("predict speed {} s".format(time.time() - start_time))

            key = get_key(max_index)
            key, limit = self.__speed_controller(key)

            if limit:
                for _ in range(5):
                    press(key)

            press(key)

    def __speed_controller(self, key):
        limit = False
        if self.last_s_distance >= 5:
            if 'w' in key:
                print("limit speed")
                key = 's'
                limit = True

        if 's' in key:
            self.last_s_distance = 0
        else:
            self.last_s_distance += 1

        return key, limit

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
