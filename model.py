import tensorflow as tf
import utils


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


def build_image_cnn(avg=True):
    inputs = tf.keras.layers.Input(shape=[None, None, None, 3])

    layer = td_conv_bn(inputs, 24, 5, 2)
    layer = time_distributed(
        layer,
        tf.keras.layers.LeakyReLU()
    )

    layer = td_conv_bn(layer, 36, 3, 2)
    layer = time_distributed(
        layer,
        tf.keras.layers.LeakyReLU()
    )

    layer = td_conv_bn(layer, 48, 3, 2)
    layer = time_distributed(
        layer,
        tf.keras.layers.LeakyReLU()
    )

    layer = td_conv_bn(layer, 64, 3, 2)
    layer = time_distributed(
        layer,
        tf.keras.layers.LeakyReLU()
    )

    layer = td_conv_bn(layer, 64, 3, 2)
    layer = time_distributed(
        layer,
        tf.keras.layers.LeakyReLU()
    )

    features = time_distributed(
        layer,
        tf.keras.layers.Dropout(rate=0.2)
    )

    if avg:
        features = time_distributed(
            features,
            tf.keras.layers.GlobalAveragePooling2D()
        )

    return inputs, features


def build_image_lstm(features):
    lstm = tf.keras.layers.LSTM(50)(features)
    return lstm


def build_map_cnn(avg=True):
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    layer = conv_bn(inputs, 24, 5, 2)
    layer = conv_bn(layer, 36, 3, 2)
    layer = conv_bn(layer, 36, 3, 2)
    layer = conv_bn(layer, 50, 3, 2)

    output = tf.keras.layers.Dropout(rate=0.2)(layer)
    if avg:
        output = tf.keras.layers.GlobalAveragePooling2D()(output)

    return inputs, output


def tail_layer(image_last_layer, map_last_layer):
    combined_layer = tf.keras.layers.Concatenate()([image_last_layer, map_last_layer])
    output = tf.keras.layers.Dense(utils.n_classes)(combined_layer)

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
