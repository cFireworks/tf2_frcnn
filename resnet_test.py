import tensorflow as tf
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.preprocessing import image
from BaseNet import resnet as nn
import os

base_dir = 'E:/Workspace/Keras/keras-vgg/model_animal_data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# make the dataset, use imagedatagenerator
train_datagen = image.ImageDataGenerator(rescale=1./255)
validation_datagen = image.ImageDataGenerator(rescale=1./255)
test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

test_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# construct the model
model_path = 'E:/Workspace/Keras/keras_frcnn/model/base_net/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
img_input = Input(shape=(256, 256, 3))

base_layers = nn.resnet_base(img_input, trainable=True)
res_base = nn.resnet_last(base_layers, trainable=True)

out = Flatten()(res_base)
out_class = Dense(10, activation='softmax', kernel_initializer='zero')(out)

model_classifier = Model(img_input, out_class)
model_classifier.load_weights(model_path, by_name=True)

optimizer_classifier = Adam(lr=1e-5)
model_classifier.compile(loss='categorical_crossentropy',
                         optimizer=optimizer_classifier,
                         metrics=['acc'])

history = model_classifier.fit_generator(
    train_generator,
    steps_per_epoch=5,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=50
)
