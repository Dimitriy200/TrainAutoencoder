# import glob
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import datetime
# import keras
# import tensorflow
# import os

from numpy import genfromtxt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.datasets import make_classification
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# import glob
import numpy as np
# import plotly.express as px
# import plotly.graph_objs as go
# import plotly as pl

import tensorflow as tf
import datetime
# import keras
# import os

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Autoencoder_Model():

    def __init__(self) -> None:
        #%load_ext tensorboard
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        

        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir = log_dir, 
        #                                                    histogram_freq = 1, 
        #                                                    profile_batch = (10, 100))


    @classmethod
    def start_all_processes(self,
                            path_Train_data: str,
                            path_Valid_Data: str,
                            path_Predict_Data: str #Должно быть 2 строки данныъ. Первую можно заполнить нулями
                            ):
        
        Train_data = self.get_np_arr_from_csv(path_Train_data)
        Valid_Data = self.get_np_arr_from_csv(path_Valid_Data)
        Predict_data = self.get_np_arr_from_csv(path_Predict_Data)

        new_model = self.create_default_model()

        new_train_model = self.start_train(new_model,
                                           Train_data,
                                           Valid_Data)
        
        restauriert_data = self.start_predict_model(new_train_model, Predict_data)
        
        metrics = self.start_active_validate(restauriert_data,
                                             Predict_data)

        return  metrics



    @classmethod
    def create_default_model(self,
                             input_dim: int = 26) -> keras.Model:

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
                            metrics = ["acc"])

        autoencoder_compressing.summary()
        
        return autoencoder_compressing



    @classmethod
    def start_train(self,
                    model: keras.Model,
                    train_data: np.array,
                    valid_data: np.array,
                    epochs = 5,
                    batch_size = 150,) -> keras.Model:
        
        status_log = ["Train successfull", "Train error"]
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        log_dir = "content/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                              histogram_freq=1, 
                                                              profile_batch = (10,100))
        
        history = model.fit(
            train_data, valid_data,
            shuffle = True,
            epochs = epochs,
            batch_size = batch_size,
            callbacks = [tensorboard_callback],
            validation_data=(valid_data, valid_data))

        return model

    

    @classmethod
    def start_predict_model(self, model: keras.Model,
                            Predict_data: np.array,
                            batch_size: int = 200) -> np.array:

        return model.predict(Predict_data)
        


    @classmethod
    def start_active_validate(self,
                        x_x: np.array,
                        y_y: np.array,
                        # log_dir = os.path.join("content/logs/", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # logs for tensoboaed
                        ) -> dict :
        
        res = {}

        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
        #                                                       histogram_freq=1,
        #                                                       profile_batch = (10,100))

        MSE = keras.losses.mean_squared_error(x_x, y_y)
        RMSE = keras.losses.mean_squared_error(x_x, y_y)
        # RMSE = keras.metrics.RootMeanSquaredError()
        # RMSE.update_state(x_x, y_y)

        res["MSE"] = MSE
        res["RMSE"] = RMSE
        
        return res



    @classmethod
    def check_loss(self, inp_DF, res_DF):
        MSE     = keras.losses.mean_squared_error(inp_DF, res_DF)
        RMSE    = keras.metrics.RootMeanSquaredError()
        R2      = keras.metrics.MeanSquaredLogarithmicError()
        MAPE    = keras.metrics.R2Score()

        RMSE.update_state(inp_DF, res_DF)
        R2.update_state(inp_DF, res_DF)
        MAPE.update_state(inp_DF, res_DF)

        print("MSE = ", MSE, "\nRMSE = ", RMSE.result(), "\nR2 = ", R2.result(), "\nMAPE = ", MAPE.result())



    @classmethod
    def start_static_validate(self, 
                              model: keras.models.Model,
                              x_x: np.array,
                              y_y: np.array,):
        
        res = 0

        return res



    @classmethod
    def save_model(self, model: keras.models.Model,
                   save_filepath: str):
        
        keras.saving.save_model(model,
                                save_filepath)
        
    

    @classmethod
    def load_model(self, load_filepath: str) -> keras.models.Model:
        new_model = keras.saving.load_model(load_filepath,
                                            custom_objects=None,
                                            compile=True,
                                            safe_mode=True)

        return new_model



    @classmethod
    def get_np_arr_from_csv(self, path_cfv: str) -> np.array:
        res = genfromtxt(path_cfv, delimiter=',')
        return res