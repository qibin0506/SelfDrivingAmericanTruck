import os, csv, threading, time


class RecordTape(object):

    def __init__(self,
                 real_exit,
                 save_dir,
                 region,
                 image_box,
                 map_box):
        self.real_exit = real_exit
        self.save_dir = save_dir
        self.region = region
        self.image_box = image_box
        self.map_box = map_box

        self.tape = []
        self.lock = threading.Lock()
        self.exiting = False
        self.image_index = 0

        self.image_dir = "{}images/".format(save_dir)
        self.map_dir = "{}maps/".format(save_dir)

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        if not os.path.exists(self.map_dir):
            os.makedirs(self.map_dir)

        csv_file_name = "{}record_{}.csv".format(save_dir, int(time.time()))
        while os.path.exists(csv_file_name):
            csv_file_name = "{}record_{}.csv".format(save_dir, int(time.time()))

        self.csv_file = open(csv_file_name, "w")
        self.writer = csv.writer(self.csv_file)

        threading.Thread(target=self.run).start()

    def write(self, img, key):
        self.tape.append((img, key))

    def exit(self):
        self.exiting = True

    def run(self):
        while True:
            if len(self.tape) == 0:
                if self.exiting:
                    self.real_exit(self.csv_file)
                    break

                time.sleep(1)
            else:
                if not self.exiting:
                    time.sleep(1)

                img, key = self.tape.pop(0)
                if not os.path.exists("{}/preview.jpg".format(self.save_dir)):
                    img.save("{}/preview.jpg".format(self.save_dir))

                image, map = self.__split_and_save(img)
                name = "{}_{}".format(int(time.time()), self.image_index)
                self.image_index += 1

                image_name = "{}{}.jpg".format(self.image_dir, name)
                map_name = "{}{}.jpg".format(self.map_dir, name)

                image.save(image_name)
                map.save(map_name)

                self.writer.writerow([name, key])

    def __split_and_save(self, img):
        image = img.crop(self.image_box)
        map = img.crop(self.map_box)

        return image, map
