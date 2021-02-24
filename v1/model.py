import tensorflow as tf
import utils


"""
class WeightedAdd(tf.keras.layers.Add):
    def __init__(self, weights=None):
        super(WeightedAdd, self).__init__()
        self.add_op_weights = weights

    def __get_weight(self, index):
        if self.add_op_weights is None:
            return 1

        if index >= len(self.add_op_weights):
            return 1

        return self.add_op_weights[index]

    def _merge_function(self, inputs):
        output = tf.math.multiply(self.__get_weight(0), inputs[0])
        for i in range(1, len(inputs)):
            output += self.__get_weight(i) * inputs[i]
        return output

    # def compute_output_shape(self, input_shape):
    #     return super().compute_output_shape(input_shape)
"""

class WeightedAdd(tf.keras.layers.Layer):
    def __init__(self, weight):
        self.weight = weight
        super(WeightedAdd, self).__init__()

    def call(self, inputs, **kwargs):
        return self.weight * inputs[0] + (1. - self.weight) * inputs[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]



# def test_weighted_add():
#     a = tf.keras.layers.Input(shape=[None, None, 3])
#     b = tf.keras.layers.Input(shape=[None, None, 3])
#
#     dense_a = tf.keras.layers.Dense(1, use_bias=False)(a)
#     dense_b = tf.keras.layers.Dense(1, use_bias=False)(b)
#
#     rst = WeightedAdd(1.)([dense_a, dense_b])
#
#     model = tf.keras.models.Model(inputs=[a, b], outputs=rst)
#     loss_obj = tf.keras.losses.MeanSquaredError()
#     optimizer = tf.keras.optimizers.Adam(lr=0.01)
#
#     for _ in range(1):
#         with tf.GradientTape() as tape:
#             img = get_image("1610168825_1")
#             pred = model([[img], [img]])
#             loss = loss_obj(y_true=1, y_pred=pred)
#             print(loss)
#
#             print(model.trainable_variables)
#
#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#             print(model.trainable_variables)


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
    # f(x, y) = 0.8 * x + (1. - 0.8) * y
    combined_layer = WeightedAdd(utils.weights_of_image_branch)([image_last_layer, map_last_layer])
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
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(model, show_shapes=True, to_file='model.png')
    except:
        pass
