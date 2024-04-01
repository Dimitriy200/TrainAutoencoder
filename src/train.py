import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tensorflow.keras import layers
from tensorflow.keras import regularizers



class Autoencoder_Model():


    def __init__(self, tensorboard_callback) -> None:
        #%load_ext tensorboard
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        log_dir = "content/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                                            log_dir = log_dir, 
                                                            histogram_freq = 1, 
                                                            profile_batch = (10, 100))
        
        self.status: bool


    def createModel(self,
                    imput_data: pd.DataFrame, 
                    save_filepath) -> bool:

        status_log = ["Create model has successfull", "Create model has error"]

        try:
            input_dim = len(imput_data)
            autoencoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )), 
            tf.keras.layers.Dense(18, activation='elu'),
            tf.keras.layers.Dense(12, activation='elu'),
            tf.keras.layers.Dense(6, activation='elu'),
            tf.keras.layers.Dense(2, activation='elu'),
            
            # reconstruction / decode
            tf.keras.layers.Dense(6, activation='elu'),
            tf.keras.layers.Dense(12, activation='elu'),
            tf.keras.layers.Dense(18, activation='elu'),
            tf.keras.layers.Dense(input_dim, activation='elu')])

            autoencoder.compile(optimizer = "adam", 
                                    loss = "mse",
                                    metrics = ["acc"])
            
            tf.keras.models.save_model(autoencoder,
                                    save_filepath)
            
            print(status_log[0])
            self.status = True
            return self.status
        
        except[]:
            print(status_log[1])
            self.status = False
            return self.status 
    

    def start_train(self,
                    autoencoder, 
                    train_data, 
                    test_data, 
                    valid_data, 
                    save_filepath) -> bool:
        
        status_log = ["Train successfull", "Train error"]
        
        try:
            history = autoencoder.fit(
                train_data, test_data,
                shuffle = True,
                epochs = 150,
                batch_size = 200,
                callbacks = [self.tensorboard_callback],
                validation_data=(valid_data, valid_data))
            
            tf.keras.models.save_model(autoencoder,
                                    save_filepath)
            
            print(status_log[0])
            self.status = True
            return self.status
        
        except[]:
            print(status_log[1])
            self.status = False
            return self.status