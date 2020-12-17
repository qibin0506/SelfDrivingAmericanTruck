import numpy as np
import cv2
import csv
import utils
from keys import get_encoded_key


def get_image(name):
    img = cv2.imread("./data/images/{}.jpg".format(name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (utils.image_width, utils.image_height))
    img = img / 127.5 - 1.0

    return img


def get_map(name):
    map = cv2.imread("./data/maps/{}.jpg".format(name))
    map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
    map = cv2.resize(map, (utils.map_width, utils.map_height))
    map = map / 127.5 - 1.0

    return map


def get_batch_fn(batch_size):
    full_data = []
    with open('./data/record.csv') as f:
        reader = csv.reader(f)
        line_count = 0

        for line in reader:
            line_count += 1
            if line_count < utils.image_seq_size:
                continue

            full_data.append((line[0], line[1]))

    data_size = len(full_data)

    def batch_fn():
        for i in range(0, data_size, batch_size):
            images = []
            maps = []
            keys = []

            for data in full_data[i:i+batch_size]:
                key_encode = get_encoded_key(data[1])
                image_seq = []
                for j in range(utils.image_seq_size):
                    image_seq.append(get_image(data[0]))

                map = get_map(data[0])

                images.append(image_seq)
                maps.append(map)
                keys.append(key_encode)

            yield np.array(images), np.array(maps), np.array(keys)

    return batch_fn


if __name__ == '__main__':
    batch_fn = get_batch_fn(32)
    for i, m, k in batch_fn():
        print(i.shape, m.shape, k.shape)