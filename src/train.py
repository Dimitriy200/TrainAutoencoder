import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import keras
import tensorflow
import os

from numpy import genfromtxt
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
    def create_default_model(self, input_dim: int = 26) -> keras.Model:

        status_log = ["Create model has successfull", "Create model has error"]

        autoencoder_compressing = keras.models.Sequential([
            keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )),
            keras.layers.Dense(16, activation='elu'),

            keras.layers.Dense(10, activation='elu'),
            
            keras.layers.Dense(16, activation='elu'),
            keras.layers.Dense(input_dim, activation='elu')
        ])

        autoencoder_compressing.compile(optimizer = "adam",
                            loss = ["mse"],
                            metrics = "acc")

        autoencoder_compressing.summary()
        
        return autoencoder_compressing



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
    def start_active_validate(self,
                        model: keras.Model,
                        x_x: np.array,
                        y_y: np.array,
                        batch_size = 200,
                        log_dir = os.path.join("content/logs/", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # logs for tensoboaed
                        ) -> dict :
        
        res = {}

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1,
                                                              profile_batch = (10,100))

        MSE = keras.losses.mean_squared_error(x_x, y_y)
        RMSE = keras.metrics.RootMeanSquaredError()
        RMSE.update_state(x_x, y_y)

        res["MSE"] = MSE
        res["RMSE"] = RMSE
        
        return res



    @classmethod
    def start_static_validate(self, 
                              model: keras.models.Model,
                              x_x: np.array,
                              y_y: np.array,):
        
        res = 0

        return res



    @classmethod
    def save_model(model: keras.models.Model,
                   save_filepath: str):
        
        keras.saving.save_model(model,
                                save_filepath)
        
    

    @classmethod
    def load_model(load_filepath: str) -> keras.models.Model:
        new_model = keras.saving.load_model(load_filepath,
                                            custom_objects=None,
                                            compile=True,
                                            safe_mode=True)

        return new_model



    @classmethod
    def get_np_arr_from_csv(path_cfv: str) -> np.array:
        return genfromtxt(path_cfv, delimiter=',')