image_height = 88
image_width = 402

map_height = 26
map_width = 88

image_seq_size = 5

epochs = 500
batch_size = 32

n_classes = 9

ckpt_path = './ckpt/train/'


def get_lr(epoch):
    if epoch < 2:
        return 1e-4
    else:
        return 1e-5
