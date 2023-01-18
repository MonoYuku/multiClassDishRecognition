import os
import sys
import random
import unittest
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa 
import matplotlib.pyplot as plt
from PIL import ImageFile
from tensorflow import keras
from keras.models import Sequential
from keras import layers, Input, Model
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

version=45

tf.test.gpu_device_name()
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dirs:
    train = './data/training'
    test = './data/test'
    val = './data/validation'
    diagram = './diagrams/dish_recognition'+str(version)+'.png'
    log = './logs/log'+str(version)+'.log'
    training_log = './logs/train_history' + str(version) + '.log'
    model = './models/best_dish_recognition'+str(version)+'.hdf5'


class Constants:
    batch_size = 32
    img_height = 299
    img_width = 299
    epochs = 100
    seed = 45
    lr = 0.0001
    momentum = 0.9
    verbose = 1
    patience = 10


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model(constant):
    set_seed(constant.seed)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    for layer in base_model.layers[:249]:
        layer.trainable = False
    for layer in base_model.layers[249:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(22, kernel_regularizer=regularizers.l2(0.005),activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    #f1 = tfa.metrics.F1Score(num_classes=10,average='weighted')
    #mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=10)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=constant.lr, momentum=constant.momentum), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_callbacks(constant,dirs):
    checkpointer = ModelCheckpoint(filepath=dirs.model, verbose=constant.verbose, save_best_only=True)
    csv_logger = CSVLogger(dirs.training_log)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', restore_best_weights=True, verbose=constant.verbose, patience=constant.patience)
    return [checkpointer,csv_logger,callback]


def get_data(constant, dirs):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.85,1.15]
        )

    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = datagen.flow_from_directory(
        dirs.train,
        target_size=(constant.img_height, constant.img_width),
        batch_size=constant.batch_size,
        class_mode='categorical',
        shuffle=True) 

    test_generator = test_datagen.flow_from_directory(
        dirs.test,
        target_size=(constant.img_height, constant.img_width),
        batch_size=constant.batch_size,
        class_mode='categorical',
        shuffle=False) 

    validation_generator = test_datagen.flow_from_directory(
        dirs.val,
        target_size=(constant.img_height, constant.img_width),
        batch_size=constant.batch_size,
        class_mode='categorical',
        shuffle=False) 
    
    return train_generator, test_generator, validation_generator
    
def get_weights(d):
    rootdir = d.train
    classes=[]
    for subdir, dirs, files in os.walk(rootdir):
        i = 1
        for file in files:
            i+=1
        if(i>1):
            classes.append(i)
    class_weight={}
    i=0
    for c in classes:
        class_weight[i]=max(classes)/c
        i+=1
    return class_weight


def get_plot(history,dirs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axhline(y=0.13349126,label='most sapmled class chance', color="teal",linestyle="dotted")
    plt.axhline(y=0.300947558,label='top3 most sampled classes', color="turquoise",linestyle="dotted")
    plt.axhline(y=0.43349126,label='top5 most sampled classes', color="cadetblue",linestyle="dotted")
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,max(plt.ylim())])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(dirs.diagram)
    #plt.show()


if __name__ == '__main__':
    dirs = Dirs()
    constant = Constants()

    f = open(dirs.log, 'w')
    sys.stdout = f

    model = get_model(constant)

    train_dataset, test_dataset, validation_dataset = get_data(constant,dirs)

    history = model.fit(
    train_dataset,
    steps_per_epoch = train_dataset.samples // constant.batch_size,
    validation_data = test_dataset,
    validation_steps = test_dataset.samples // constant.batch_size,
    epochs = constant.epochs,
    verbose=constant.verbose,
    callbacks=get_callbacks(constant,dirs),
    class_weight=get_weights(dirs)
    )

    get_plot(history, dirs)



    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)

    loss, accuracy = model.evaluate(validation_dataset)
    print('Validation accuracy :', accuracy)
    f.close()
