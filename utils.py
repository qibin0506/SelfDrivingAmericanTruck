image_height = 177
image_width = 803

map_height = 51
map_width = 177

image_seq_size = 5

epochs = 50
batch_size = 16

n_classes = 9

ckpt_path = './ckpt/train/'


def get_lr(epoch):
    if epoch < 2:
        return 1e-5
    else:
        return 1e-6
