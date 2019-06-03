from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from Configs import config
from Detection import losses as losses_fn
from Detection import detect_layer
from BaseNet import resnet as nn


# config for data argument
cfg = config.Config()
cfg.model_path = 'E:/Workspace/Keras/keras_frcnn/model/base_net/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

input_shape_img = (None, None, 3)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))
# define the base network (resnet here conv1-conv4, can be VGG, Inception, etc)
shared_layers = nn.resnet_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
rpn = detect_layer.rpn(shared_layers, num_anchors)

# the head of the detection-framework, detect each ROI with roipooling and conv5
classifier = detect_layer.head(shared_layers, roi_input, cfg.num_rois, nb_classes=10, trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
    model_rpn.load_weights(cfg.model_path, by_name=True)
    model_classifier.load_weights(cfg.model_path, by_name=True)
except Exception as e:
    print(e)
    print('Could not load pretrained model weights. Weights can be found in the keras application folder '
          'https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer,
                  loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier,
                         loss=[losses_fn.head_loss_cls, losses_fn.head_loss_regr(10 - 1)],
                         metrics={'dense_class_{}'.format(10): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

