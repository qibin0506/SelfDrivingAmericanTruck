import numpy as np
import cv2
import csv
import random
import utils
from glob import glob
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


def read_cvs(file):
    full_data = []
    with open(file) as f:
        reader = csv.reader(f)

        for line in reader:
            full_data.append((line[0], line[1]))

    return full_data


def data_analysis(data):
    rst = {
        'w': 0,
        's': 0,
        'a': 0,
        'd': 0,
        'wa': 0,
        'wd': 0,
        'sa': 0,
        'sd': 0,
        'no': 0
    }

    for line in data:
        key = line[1]

        if key == 'w':
            rst['w'] += 1
        elif key == 's':
            rst['s'] += 1
        elif key == 'a':
            rst['a'] += 1
        elif key == 'd':
            rst['d'] += 1
        elif key == 'wa' or key == 'aw':
            rst['wa'] += 1
        elif key == 'wd' or key == 'dw':
            rst['wd'] += 1
        elif key == 'sa' or key == 'as':
            rst['sa'] += 1
        elif key == 'sd' or key == 'ds':
            rst['sd'] += 1
        elif key == 'no':
            rst['no'] += 1

    print("before balance, data analysis: {}, total: {}".format(rst, sum(rst.values())))

    balance_size = min(rst['w'], rst['wa'], rst['wd'], rst['no'])

    rst['balance_w'] = min(balance_size + 500, rst['w'])
    rst['balance_wa'] = min(balance_size + 200, rst['wa'])
    rst['balance_wd'] = min(balance_size + 200, rst['wd'])
    rst['balance_no'] = min(balance_size, rst['no'])

    return rst


def data_balance(data, analysis_map):
    cur_data_size = len(data)
    pass_list = [False for _ in range(cur_data_size)]

    final_w_count = 0
    final_wa_count = 0
    final_wd_count = 0
    final_no_count = 0

    for index in range(cur_data_size):
        key = data[index][1]
        pass_data = False

        if key == 'w':
            pass_data = random.random() > analysis_map['balance_w'] / analysis_map['w']
            if not pass_data:
                final_w_count += 1
        elif key == 'wa' or key == 'aw':
            pass_data = random.random() > analysis_map['balance_wa'] / analysis_map['wa']
            if not pass_data:
                final_wa_count += 1
        elif key == 'wd' or key == 'dw':
            pass_data = random.random() > analysis_map['balance_wd'] / analysis_map['wd']
            if not pass_data:
                final_wd_count += 1
        elif key == 'no':
            pass_data = random.random() > analysis_map['balance_no'] / analysis_map['no']
            if not pass_data:
                final_no_count += 1

        pass_list[index] = pass_data

    analysis_map['w'] = final_w_count
    analysis_map['wa'] = final_wa_count
    analysis_map['wd'] = final_wd_count
    analysis_map['no'] = final_no_count

    del analysis_map['balance_w']
    del analysis_map['balance_wa']
    del analysis_map['balance_wd']
    del analysis_map['balance_no']

    print("after balance, data analysis: {}, total: {}".format(analysis_map, sum(analysis_map.values())))

    return pass_list


def get_batch_fn(batch_size):
    records = glob('./data/*.csv')
    if random.randint(0, 9) >= 5:
        records.reverse()

    all_data = []
    analysis_maps = []

    for record in records:
        cur_data = read_cvs(record)
        all_data.append(cur_data)

        analysis_maps.append(data_analysis(cur_data))

    def on_epoch(epoch):
        all_data_size = len(all_data)
        pass_lists = []
        final_count = 0

        for idx in range(all_data_size):
            data = all_data[idx]
            analysis_map = analysis_maps[idx].copy()

            pass_lists.append(data_balance(data, analysis_map))
            final_count += sum(analysis_map.values())

        def batch_fn():
            images = []
            maps = []
            keys = []

            for data_idx in range(all_data_size):
                data = all_data[data_idx]
                data_size = len(data)
                pass_list = pass_lists[data_idx]

                for idx in range(data_size):
                    if idx < utils.image_seq_size - 1:
                        continue

                    if pass_list[idx]:
                        continue

                    line = data[idx]
                    key_encode = get_encoded_key(line[1])
                    map = get_map(line[0])
                    image_seq = []

                    for k in range(utils.image_seq_size - 1, -1, -1):
                        image_seq.append(get_image(data[idx - k][0]))

                    images.append(image_seq)
                    maps.append(map)
                    keys.append(key_encode)

                    if len(images) == batch_size:
                        yield shuffle(np.array(images), np.array(maps), np.array(keys))

                        images = []
                        maps = []
                        keys = []

                if len(images) != 0:
                    yield shuffle(np.array(images), np.array(maps), np.array(keys))

                    images = []
                    maps = []
                    keys = []

        return batch_fn, final_count // batch_size

    return on_epoch


if __name__ == '__main__':
    epoch_fn = get_batch_fn(32)
    batch_fn, count = epoch_fn(0)
    for i, m, k in batch_fn():
        print(i.shape, m.shape, k.shape, count)
