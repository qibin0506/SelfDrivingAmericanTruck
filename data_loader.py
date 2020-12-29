import numpy as np
import cv2
import csv
import random
import utils
from keys import get_encoded_key
from sklearn.utils import shuffle


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


def read_cvs(file='./data/record.csv'):
    full_data = []
    with open(file) as f:
        reader = csv.reader(f)

        for line in reader:
            full_data.append((line[0], line[1]))

    return full_data


def data_analysis(data):
    w_count = 0
    s_count = 0
    a_count = 0
    d_count = 0
    wa_count = 0
    wd_count = 0
    sa_count = 0
    sd_count = 0
    no_count = 0

    for line in data:
        key = line[1]

        if key == 'w':
            w_count += 1
        elif key == 's':
            s_count += 1
        elif key == 'a':
            a_count += 1
        elif key == 'd':
            d_count += 1
        elif key == 'wa' or key == 'aw':
            wa_count += 1
        elif key == 'wd' or key == 'dw':
            wd_count += 1
        elif key == 'sa' or key == 'as':
            sa_count += 1
        elif key == 'sd' or key == 'ds':
            sd_count += 1
        elif key == 'no':
            no_count += 1

    return w_count, s_count, a_count, d_count, wa_count, wd_count, sa_count, sd_count, no_count


def get_batch_fn(batch_size):
    full_data = read_cvs()

    data_size = len(full_data)

    w_count, s_count, a_count, d_count, wa_count, wd_count, sa_count, sd_count, no_count = data_analysis(full_data)
    print("data analysis: w: {}, s: {}, a: {}, d: {}, wa: {}, wd: {}, sa: {}, sd: {}, no: {}".
          format(w_count, s_count, a_count, d_count, wa_count, wd_count, sa_count, sd_count, no_count))

    def batch_fn():
        i_list = list(range(utils.image_seq_size - 1, data_size, batch_size))

        for i in i_list:
            images = []
            maps = []
            keys = []

            for j in range(i, i+batch_size):
                data = full_data[j]

                key_encode = get_encoded_key(data[1])
                image_seq = []
                for k in range(utils.image_seq_size - 1, -1, -1):
                    image_seq.append(get_image(full_data[j-k][0]))

                map = get_map(data[0])

                images.append(image_seq)
                maps.append(map)
                keys.append(key_encode)

            yield shuffle(np.array(images), np.array(maps), np.array(keys))

    return batch_fn, data_size // batch_size


if __name__ == '__main__':
    batch_fn, _ = get_batch_fn(2)
    for i, m, k in batch_fn():
        print(i.shape, m.shape, k.shape)
