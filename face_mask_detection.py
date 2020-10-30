import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Input, BatchNormalization
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from util import *
from variables import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class FaceMaskModel(object):
    def __init__(self):
        train_generator, validation_generator, test_generator = image_data_generator()
        self.test_generator = test_generator
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.train_step = self.train_generator.samples // batch_size
        self.validation_step = self.validation_generator.samples // valid_size
        self.test_step = self.test_generator.samples // batch_size

        class_dict = self.train_generator.class_indices
        self.class_dict = {v:k for k,v in class_dict.items()} #{0: 'with_mask', 1: 'without_mask'}

    def model_conversion(self):
        num_classes = len(get_class_labels())
        mobilenet_functional = tf.keras.applications.MobileNet()
        model = Sequential()
        for layer in mobilenet_functional.layers[:-1]:
            layer.trainable = False
            model.add(layer)
        model.add(Dense(dense_1, activation='relu'))
        model.add(Dense(dense_1, activation='relu')) 
        model.add(BatchNormalization()) 
        model.add(Dense(dense_2, activation='relu'))
        model.add(Dense(dense_2, activation='relu'))
        model.add(BatchNormalization()) 
        model.add(Dense(dense_3, activation='relu'))
        model.add(Dense(dense_3, activation='relu'))
        model.add(Dense(dense_3, activation='relu'))
        model.add(Dense(num_classes, activation='sigmoid'))
        model.summary()
        self.model = model

    def train(self):
        self.model.compile(
                          optimizer='Adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )
        self.model.fit_generator(
                          self.train_generator,
                          steps_per_epoch=self.train_step,
                          validation_data=self.validation_generator,
                          validation_steps=self.validation_step,
                          epochs=epochs,
                          verbose=verbose
                        )

    def save_model(self):
        print("Drowsiness model Saving !")
        model_json = self.model.to_json()
        with open(model_architecture, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)

    def load_model(self):
        K.clear_session() #clearing the keras session before load model
        json_file = open(model_architecture, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(model_weights)

        self.model.compile(
                          optimizer='Adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )

        print("Drowsiness Model Loaded")

    def evaluation(self):
        Predictions = self.model.predict_generator(self.test_generator,steps=self.test_step)
        P = np.argmax(Predictions,axis=1)
        loss , accuracy = self.model.evaluate_generator(self.test_generator, steps=self.test_step)
        print("test loss : ",loss)
        print("test accuracy : ",accuracy)

    def predition(self, img):
        Prediction = self.model.predict_generator(img)
        P = int(np.argmax(Prediction,axis=1).squeeze())
        Pclass = self.class_dict[P]
        return Pclass

    def run(self):
        if os.path.exists(model_weights):
            self.load_model()
        else:
            self.model_conversion()
            self.train()
            self.save_model()
        # self.evaluation()

# if __name__ == "__main__":
#     model = FaceMaskModel()
#     model.run()