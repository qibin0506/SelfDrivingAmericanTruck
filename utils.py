default_region = [494, 190, 808, 507] # left, top, width, height
default_image_box = [0, 272, 1606, 626] # left, top, right, bottom
default_map_box = [1218, 752, 1572, 854] # left, top, right, bottom

image_height = 66
image_width = 300

map_height = 20
map_width = 66

image_seq_size = 5
pred_skip_frame = 1 # 7 # train_record_time(0.2) / pred_time(0.03)

epochs = 500
batch_size = 200

n_classes = 9
weights_of_image_branch = 0.9

ckpt_path = './ckpt/train/'
model_path = 'sda_driver'


def get_lr(epoch):
    return 1e-4
    # if epoch < 10:
    #     return 1e-4
    # else:
    #     return 1e-5
