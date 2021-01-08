default_region = [494, 190, 808, 507] # left, top, width, height
default_image_box = [0, 272, 1606, 626] # left, top, right, bottom
default_map_box = [1218, 752, 1572, 854] # left, top, right, bottom

image_height = 88
image_width = 402

map_height = 26
map_width = 88

image_seq_size = 5
pred_skip_frame = 7 # train_record_time(0.2) / pred_time(0.03)

epochs = 500
batch_size = 74

n_classes = 9

ckpt_path = './ckpt/train/'


def get_lr(epoch):
    return 1e-5
    # if epoch < 10:
    #     return 1e-4
    # else:
    #     return 1e-5
