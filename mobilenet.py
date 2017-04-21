"""
Modules implementing variants of MobileNet

https://arxiv.org/pdf/1704.04861.pdf
"""

import tensorflow as tf
import sonnet as snt


def global_average_pool(input_fmaps):
    """Performs global average pooling."""
    if not len(input_fmaps.get_shape()) == 4:
        raise ValueError('Global average pool expects 4 dimensions, '
                         'got shape {}'.format(fmaps.get_shape))
    return tf.reduce_mean(input_fmaps, axis=[1, 2], name='avge_pool')


class MobileNet(snt.AbstractModule):
    """Core MobileNet architecture, aimed towards classification.

    Conv 3x3x3x32 (stride 2x2)
    Conv 3x3x32 (depthwise)
    Conv 1x1x32x64
    Conv 3x3x64 (depthwise) (stride 2x2)
    Conv 1x1x64x128
    Conv 3x3x128 (depthwise)
    Conv 1x1x128x128
    Conv 3x3x128 (depthwise) (stride 2x2)
    Conv 1x1x128x256
    Conv 3x3x256 (depthwise)
    Conv 1x1x256x256
    Conv 3x3x256 (depthwise) (stride 2x2)
    Conv 1x1x256x512
    Conv 3x3x512 (depthwise)
    Conv 1x1x512x512
    Conv 3x3x512 (depthwise)
    Conv 1x1x512x512
    Conv 3x3x512 (depthwise)
    Conv 1x1x512x512
    Conv 3x3x512 (depthwise)
    Conv 1x1x512x512
    Conv 3x3x512 (depthwise)
    Conv 1x1x512x512
    Conv 3x3x512 (depthwise) (stride 2x2)
    Conv 1x1x512x1024
    Conv 3x3x1024 (depthwise)
    Conv 1x1x1024x1024
    Global average pool
    FC 1024xnum_classes
    Softmax
    """

    # TODO: regularisers
    def _depthwise_separable_conv(self, output_fmaps, stride=1):
        """Returns a list of modules/ops implementing a single depthwise
         separable convolution, as defined in the MobileNet paper:

         3x3 depthwise, with optional stride
         batch norm
         relu
         1x1 pointwise with output_fmaps output channels
         batch norm
         relu

         Args:
            output_fmaps (int): the number of feature maps resulting from the
                1x1 convolution.
            stride (Optional[int]): stride, applied evenly in both dimensions.
                Defaults to 1.

        Returns:
            list of modules/ops as above.
        """
        modules = [
            snt.DepthwiseConv2D(1, [3, 3], stride=stride, use_bias=False),
            snt.BatchNorm(),
            tf.nn.relu,
            snt.Conv2D(output_fmaps, [1, 1], use_bias=False),
            snt.BatchNorm(),
            tf.nn.relu]
        return modules

    def _scale_fmaps(self, base_width):
        """scale a number of feature maps according to the width parameter."""
        return int(self._alpha * base_width)

    def _create_submodules(self):
        """Create a submodule which, when built, will create all of the
        graph required. This is where the actual net is defined."""
        # 1 layer
        modules = [
            snt.Conv2D(self._scale_fmaps(32), [3, 3], stride=2,
                       use_bias=False),
            snt.BatchNorm(),
            tf.nn.relu]
        # 3 layers
        modules.extend(self._depthwise_separable_conv(self._scale_fmaps(64)))
        # 5 layers
        modules.extend(self._depthwise_separable_conv(self._scale_fmaps(128),
                                                      stride=2))
        # 7 layers
        modules.extend(self._depthwise_separable_conv(self._scale_fmaps(128)))
        # 9 layers
        modules.extend(self._depthwise_separable_conv(self._scale_fmaps(256),
                                                      stride=2))
        # 11 layers
        modules.extend(self._depthwise_separable_conv(self._scale_fmaps(256)))
        # 13 layers
        modules.extend(self._depthwise_separable_conv(self._scale_fmaps(512),
                                                      stride=2))
        # 23 layers
        for _ in range(5):
            modules.extend(self._depthwise_separable_conv(
                self._scale_fmaps(512)))
        # 25 layers
        modules.extend(self._depthwise_separable_conv(self._scale_fmaps(1024),
                                                      stride=2))
        # 27 layers
        modules.extend(self._depthwise_separable_conv(self._scale_fmaps(1024)))

        modules.append(global_average_pool)

        # 28 layers
        modules.append(snt.Linear(self._num_outputs))

        return modules

    def __init__(self, num_outputs, alpha=1.0, depthwise_l2=0.0,
                 pointwise_l2=0.0, name='mobile_net'):
        """
        Set up the MobileNet module.

        Args:
            num_outputs (int): the number of outputs the net will have, usually
                the number of classes for classification.
            alpha (Optional[float]): width scale -- this is used the scale the
                number of feature maps at each layer. Values in the paper were
                1.0, 0.75, 0.5 and 0.25 to test restricting the amount of
                parameters/computation. Default is 1.0 for the standard net.
            depthwise_l2 (Optional[float]): amount of l2 regularisation applied
                to the depthwise filters. The original paper mentions it is
                usually preferable to use less l2 on these filters than on the
                pointwise ones as they constitute a much smaller set of
                parameters.
            pointwise_l2 (Optional[float]): amount of l2 regularisation applied
                to the pointwise filters.
        """
        super(MobileNet, self).__init__(name=name)
        self._num_outputs = num_outputs
        self._alpha = alpha
        self._pointwise_l2 = pointwise_l2
        self._depthwise_l2 = depthwise_l2

        # construct all the modules we need
        with self._enter_variable_scope():
            self._submodules = self._create_submodules()

    def _build(self, inputs, is_training):
        """Compute output logits from input images.

        Args:
            inputs (tensor): input images, need to be
                `[batch, width, height, channels]`
            is_training (bool): whether to use moving average or batch
                statistics for the batch norm.

        Returns:
            logits for the net

        Raises:
            ValueError: if the inputs are not the correct rank.
        """
        if len(inputs.get_shape()) != 4:
            raise ValueError('Expected inputs with 4 dimensions, '
                             'got {}'.format(inputs.get_shape()))
        net_outputs = inputs
        for module in self._submodules:
            if isinstance(module, snt.BatchNorm):
                net_outputs = module(net_outputs, is_training)
            else:
                net_outputs = module(net_outputs)
            tf.logging.debug('%s, %s',
                             module,
                             net_outputs.get_shape())
        return net_outputs
