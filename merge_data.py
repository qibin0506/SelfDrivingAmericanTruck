import os
from glob import glob

from_dir = "./data_1/"
to_dir ="./data/"

from_image_dir = "{}images/*".format(from_dir)
to_image_dir = "{}images/".format(to_dir)
from_map_dir = "{}maps/*".format(from_dir)
to_map_dir = "{}maps/".format(to_dir)

images = glob(from_image_dir)
maps = glob(from_map_dir)

for img in images:
    img_name = img.split("/")[-1]
    os.rename(img, "{}{}".format(to_image_dir, img_name))

for map in maps:
    map_name = map.split("/")[-1]
    os.rename(map, "{}{}".format(to_map_dir, map_name))

csvs = glob("{}/*.csv".format(from_dir))
for csv in csvs:
    csv_name = csv.split("/")[-1]
    os.rename(csv, "{}{}".format(to_dir, csv_name))