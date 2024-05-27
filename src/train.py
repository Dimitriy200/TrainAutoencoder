import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import keras
import tensorflow


# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.datasets import make_classification
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE



class Autoencoder_Model():

    def __init__(self, tensorboard_callback) -> None:
        #%load_ext tensorboard
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        log_dir = "content/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir = log_dir, 
                                                           histogram_freq = 1, 
                                                           profile_batch = (10, 100))



    @classmethod
    def create_default_model(self,
                    input_dim: int,
                    save_filepath: str) -> keras.models.Model:

        status_log = ["Create model has successfull", "Create model has error"]

        autoencoder_compressing = keras.models.Sequential([
            keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )),
            keras.layers.Dense(24, activation='elu'),
            keras.layers.Dense(22, activation='elu'),
            keras.layers.Dense(20, activation='elu'),
            keras.layers.Dense(18, activation='elu'),
            keras.layers.Dense(16, activation='elu'),
            keras.layers.Dense(14, activation='elu'),
            keras.layers.Dense(12, activation='elu'),
            keras.layers.Dense(10, activation='elu'),

            keras.layers.Dense(10, activation='elu'),
            keras.layers.Dense(12, activation='elu'),
            keras.layers.Dense(14, activation='elu'),
            keras.layers.Dense(16, activation='elu'),
            keras.layers.Dense(18, activation='elu'),
            keras.layers.Dense(20, activation='elu'),
            keras.layers.Dense(22, activation='elu'),
            keras.layers.Dense(24, activation='elu'),
            keras.layers.Dense(input_dim, activation='elu')
        ])

        autoencoder_compressing.compile(optimizer = "adam",
                            loss = ["mse"],
                            metrics = "acc")

        autoencoder_compressing.summary()
        
        return autoencoder_compressing



    @classmethod
    def stert_validate(self,
                        model: keras.models.Model,
                        x_x: np.array,
                        y_y: np.array,
                        batch_size = 200,
                        log_dir = "content/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1,
                                                              profile_batch = (10,100))

        res = model.predict(
            x = x_x,
            y = y_y,
            batch_size = batch_size,
            callbacks = [tensorboard_callback])
        
        return res



    @classmethod
    def start_train(self,
                    model: keras.Model,
                    train_data: np.array,
                    valid_data: np.array,
                    epochs = 150,
                    batch_size = 200,) -> keras.Model:
        
        status_log = ["Train successfull", "Train error"]
        
        history = model.fit(
            train_data, valid_data,
            shuffle = True,
            epochs = epochs,
            batch_size = batch_size,
            callbacks = [self.tensorboard_callback],
            validation_data=(valid_data, valid_data))

        return model



    @classmethod
    def save_model(model: keras.models.Model,
                   save_filepath: str):
        
        keras.saving.save_model(model,
                                save_filepath)
        
    

    @classmethod
    def load_model(load_filepath: str):
        new_model = keras.saving.load_model(load_filepath,
                                            custom_objects=None,
                                            compile=True,
                                            safe_mode=True)

        return new_model