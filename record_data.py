import sys, getopt, os
import csv
import time
from pynput import keyboard

from keys import listen, get_cur_key
from cap_screen import capture
from record_tape import RecordTap


if __name__ == '__main__':
    flag = False
    exiting = False

    opts, _ = getopt.getopt(sys.argv[1:], '', ['dir=', 'region='])

    save_dir = "./data/"
    region = [318, 137, 808, 507]
    image_box = [126, 163, 910, 422]
    map_box = [879, 517, 1116, 607]

    for opt, arg in opts:
        if opt == '--dir':
            save_dir = arg
        elif opt == '--region':
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

    if not save_dir.endswith('/'):
        save_dir += '/'

    def real_exit(csv_file):
        csv_file.close()
        os._exit(0)

    record_tape = RecordTap(real_exit, save_dir, region, image_box, map_box)

    def key_event(key):
        global flag, exiting

        if key == keyboard.Key.esc:
            exiting = True
            record_tape.exit()
        elif key == keyboard.Key.tab:
            flag = True

    listen(lambda key: key_event(key))

    while not flag:
        print("waiting for start.")
        time.sleep(1)

    print("start recording.")
    while True:
        time.sleep(0.2)

        if exiting:
            print("waiting for exit.")
            continue

        img, _ = capture(region=region)
        record_tape.write(img, get_cur_key())
