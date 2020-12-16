import tensorflow as tf


def time_distributed(layer, op):
    layer = tf.keras.layers.TimeDistributed(
        op
    )(layer)

    return layer


def td_conv_bn(layer, filters, kernel_size, strides, padding='same'):
    layer = time_distributed(
        layer,
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        ))

    layer = time_distributed(
        layer,
        tf.keras.layers.BatchNormalization())

    return layer


def conv_bn(layer, filters, kernel_size, strides, padding='same'):
    layer = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        )(layer)

    layer = tf.keras.layers.BatchNormalization()(layer)

    return layer


def td_res_block(layer, filters, down_sample=False):
    shortcut = layer
    strides = 2 if down_sample else 1

    layer = td_conv_bn(layer, filters, 1, 1)
    layer = time_distributed(
        layer,
        tf.keras.layers.LeakyReLU())

    layer = td_conv_bn(layer, filters, 3, strides)
    layer = time_distributed(
        layer,
        tf.keras.layers.LeakyReLU())

    layer = td_conv_bn(layer, filters, 1, 1)

    if down_sample:
        shortcut = td_conv_bn(shortcut, filters, 1, strides)

    layer = tf.keras.layers.Add()([shortcut, layer])

    layer = time_distributed(
        layer,
        tf.keras.layers.LeakyReLU())

    return layer


def build_image_cnn():
    inputs = tf.keras.layers.Input(shape=[None, 90, 320, 3])
    layer = td_conv_bn(inputs, 32, 7, 2)

    # layer = td_res_block(layer, 32, True)
    # for _ in range(2):
    #     layer = td_res_block(layer, 32, False)

    layer = td_res_block(layer, 64, True)
    for _ in range(2):
        layer = td_res_block(layer, 64, False)

    layer = td_res_block(layer, 128, True)
    for _ in range(2):
        layer = td_res_block(layer, 128, False)

    layer = td_res_block(layer, 256, True)
    for _ in range(3):
        layer = td_res_block(layer, 256, False)

    layer = time_distributed(
        layer,
        tf.keras.layers.Dropout(rate=0.2))

    feature = time_distributed(
        layer,
        tf.keras.layers.Flatten())

    return inputs, feature


def build_image_lstm(features):
    layer = tf.keras.layers.LSTM(50)(features)
    layer_layer = tf.keras.layers.Dense(100)(layer)

    return layer_layer


def build_map_cnn():
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    layer = conv_bn(inputs, 24, 5, 2)
    layer = conv_bn(layer, 32, 5, 2)
    layer = conv_bn(layer, 64, 5, 2)
    layer = conv_bn(layer, 128, 3, 2)
    layer = conv_bn(layer, 256, 3, 2)

    layer = tf.keras.layers.Flatten()(layer)
    last_layer = tf.keras.layers.Dense(100)(layer)

    return inputs, last_layer


def tail_layer(image_last_layer, map_last_layer):
    combined_layer = tf.keras.layers.Concatenate()([image_last_layer, map_last_layer])
    layer = tf.keras.layers.Dense(50)(combined_layer)
    layer = tf.keras.layers.Dense(10)(layer)
    output = tf.keras.layers.Dense(1)(layer)

    return output


def create_model(summary=False):
    image_inputs, features = build_image_cnn()
    image_last_layer = build_image_lstm(features)

    map_inputs, map_last_layer = build_map_cnn()
    outputs = tail_layer(image_last_layer, map_last_layer)

    model = tf.keras.models.Model(inputs=[image_inputs, map_inputs], outputs=outputs)
    if summary:
        model.summary()

    return model


if __name__ == '__main__':
    model = create_model(True)
