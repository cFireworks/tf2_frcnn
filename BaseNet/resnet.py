from keras.layers import Input, Add, Activation, Conv2D, MaxPool2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed

from keras import backend as K
from BaseNet.fixed_batch_normalization import FixedBatchNormalization
bn_axis = 3


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    ''' identity_block of resnet
    input_tensor: a InputLayer class input
    kernel_size: the size of conv kernel in the 2nd layer of identity_block
    filters: [f1, f2, f3] filter_num of each layer in identity_block
    stage: the id of identity_block in a Block, to generate the name of Layers
    block: the id of Block
    trainable: param of training
    '''

    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # part 'a' of this identity block
    x = Conv2D(filters=nb_filter1, kernel_size=(1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # part 'b' of this identity block
    x = Conv2D(filters=nb_filter2, kernel_size=(kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
               trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # part 'c' of this identity block
    x = Conv2D(filters=nb_filter3, kernel_size=(1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    '''

    :param input_tensor:
    :param kernel_size:
    :param filters:
    :param stage:
    :param block:
    :param trainable:
    :return:
    '''
    # identity block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(
        Conv2D(filters=nb_filter1, kernel_size=(1, 1), trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(
        Conv2D(filters=nb_filter2, kernel_size=(kernel_size, kernel_size), trainable=trainable,
               kernel_initializer='normal',
               padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(
        Conv2D(filters=nb_filter3, kernel_size=(1, 1), trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    ''' this block is different from the paper

    :param input_tensor:
    :param kernel_size:
    :param filters:
    :param stage:
    :param block:
    :param strides:
    :param trainable:
    :return:
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(
        input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
               trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(
        input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    # conv block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(
        Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable,
                               kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c',
                        trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(
        Conv2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_base(input_tensor=None, trainable=False):
    ''' base_resnet, block 1 to 4

    :param input_tensor:
    :param trainable:
    :return:
    '''
    # Determine proper input shape
    input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = ZeroPadding2D((3, 3))(img_input)

    # print('++ zero padding: ', x)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)

    # print('+ conv 2d: ', x)
    print('NOTE: this code only support to keras 2.0.3, newest version this line will got errors. see trace back.')
    x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)

    return x


def resnet_last(x, input_shape=None, trainable=False, detection_flag=False):
    ''' the 5th Block of resnet as the classifier layer, use time distributed

    :param x:
    :param input_shape:
    :param trainable:
    :return: a feature map of base net --- resnet
    '''
    if detection_flag:
        x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2),
                          trainable=trainable)

        x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
        x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
        x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)
    else:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(2, 2),
                          trainable=trainable)

        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
        x = AveragePooling2D((7, 7))(x)

    return x
