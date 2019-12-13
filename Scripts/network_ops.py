import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import batch_norm

class ops():
    def __init__(self, training):
        self.training = training
        self.bottle_input_and_output_channel = 64
        self.bottle_intermediate_channel = 32
        self.boundary_mining_depth = 64

    def first_layer(self, inputs, scope):
        return slim.conv2d(inputs,
                           num_outputs=self.bottle_input_and_output_channel,
                           kernel_size=3,
                           stride=4,
                           scope=scope)

    def max_pooling(self, inputs):
        # downsample x2
        return slim.max_pool2d(inputs, 2)

    def batch_normalization(self, x, scope):
        return tf.cond(self.training,
                       lambda : batch_norm(inputs=x,
                                           is_training=True,
                                           reuse=None,
                                           scope=scope),
                       lambda : batch_norm(inputs=x,
                                           is_training=False,
                                           reuse=True,
                                           scope=scope))

    def conv_bn_relu(self, inputs, output_channel, scope):
        x = slim.conv2d(inputs=inputs,
                    num_outputs=output_channel,
                    kernel_size=3,
                    activation_fn=None,
                    scope=scope+'conv')
        x = self.batch_normalization(x, scope=scope+'bn')
        x = tf.nn.relu(x)
        return x

    def bottleneck_convocation(self, inputs, scope):
        '''
        :param inputs:  4d tensor with last dimension equals 64
        :return:        4d tensor with last dimension equals 64
        '''
        x = self.conv_bn_relu(inputs, self.bottle_intermediate_channel, scope=scope+'BC_1')
        x = self.conv_bn_relu(x, self.bottle_input_and_output_channel, scope=scope+'BC_2')
        block_out = x + inputs
        return block_out

    def upsample_and_add(self, x1, x2, *args):
        '''
        :param x1:      input to upsample using transposed conv
        :param x2:      input to add
        :param args:    if there are more layers to add
        :return:        a feature map with shape
                        [1, shape(x2)[1], shape(x2)[2], self.bottle_input_and_output_channel]
        '''
        resize = tf.compat.v1.image.resize_bilinear(x1, tf.shape(x2)[1:3], align_corners=True)
        deconv_output = resize + x2 + args[0] if args else resize + x2
        return deconv_output

    def layer_before_FFM(self, inputs, interpolate_target_shape, scope):
        x = slim.conv2d(inputs,
                        num_outputs=1,
                        kernel_size=3,
                        activation_fn=None,
                        scope=scope+'conv')
        x = tf.compat.v1.image.resize_bilinear(x, interpolate_target_shape, align_corners=True)
        return x

    def feature_fusion_module(self, semantic_input, boundary_feature, max_or_avg=1):
        fusion = tf.concat([semantic_input, boundary_feature],
                           axis=-1)
        x = self.conv_bn_relu(fusion, 1, scope='FFM_block')

        if max_or_avg:
            global_pooling = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
        else:
            global_pooling = tf.reduce_max(x, axis=[1, 2], keep_dims=True)

        conv1 = slim.conv2d(global_pooling,
                            num_outputs=1,
                            kernel_size=1,
                            scope='FFM_conv1')
        conv2 = slim.conv2d(conv1,
                            num_outputs=1,
                            kernel_size=1,
                            activation_fn=tf.sigmoid,
                            scope='FFM_conv2')

        ffm_out = tf.add(tf.multiply(x, conv2), x)
        return ffm_out

    def boundary_mining(self, raw_input, BAmap):
        x = tf.concat([raw_input, BAmap], axis=-1)
        x = slim.conv2d(x,
                        num_outputs=self.boundary_mining_depth,
                        kernel_size=3,
                        activation_fn=None,
                        scope='boundary_mining_conv')
        return x