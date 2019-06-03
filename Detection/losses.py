from keras import backend as K
from keras.objectives import categorical_crossentropy

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


def rpn_loss_regr(num_anchors):
    '''
    :param num_anchors:
    :return:
    '''

    def rpn_loss_regr_fixed_num(y_true, y_pred):
        '''
        :param y_true: (samples, feature_H, feature_W, 4*num_anchors_is_overlap + 4*num_anchors_reg_t)
        :param y_pred: similar to above, different in the last axis
        :return:
        '''
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')

        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    '''
    :param num_chors:
    :return:
    '''

    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return lambda_cls_class * K.sum(
            y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred, y_true[:, :, :, num_anchors:])) / K.sum(
            epsilon + y_true[:, :, :, :num_anchors])

    return rpn_loss_cls_fixed_num


def head_loss_regr(num_classes):
    '''
    :param num_classes:
    :return:
    '''

    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(
            y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :4 * num_classes])

    return class_loss_regr_fixed_num


def head_loss_cls(y_true, y_pred):
    '''
    :param y_true:
    :param y_pred:
    :return:
    '''

    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
