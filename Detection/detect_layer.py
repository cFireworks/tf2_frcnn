import tensorflow as tf
from keras.layers import Conv2D, TimeDistributed, Flatten, Dense
from keras.engine.topology import Layer
from keras import backend as K
from BaseNet.resnet import resnet_last

class RoiPoolingConv(Layer):
    ''' ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img: (1, rows, cols, channels)
        X_roi: (1,num_rois,4) list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape: (1, num_rois, pool_size, pool_size, channels)
    '''

    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        # assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # X_img: (1, rows, cols, channels)
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):

        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert (len(x) == 2)

        img = x[0]
        rois = x[1]
        outputs = []

        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            # NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
            # This code just support tensorflow
            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        # final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))  # use backend of tensorflow

        return final_output


def rpn(base_layers, num_anchors):
    '''

    :param base_layers: net for recognition
    :param num_anchors: different scales and ratios of each sliding window
    :return:
    '''
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                            name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                           name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def head(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    # https://arleyzhang.github.io/articles/7e6bc4a/

    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = resnet_last(out_roi_pool, input_shape=input_shape, trainable=True, detection_flag=True)

    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]
