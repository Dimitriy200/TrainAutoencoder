import numpy as np
import tensorflow as tf
import datetime
import mlflow
import subprocess
import dagshub


# from tensorflow import keras
# from tensorflow.python import keras
from mlflow.models import infer_signature
from numpy import genfromtxt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mlflow.models import infer_signature

# mlflow server --host 127.0.0.1 --port 5050



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
                            path_Predict_Data: str, #Должно быть 2 строки данныъ. Первую можно заполнить нулями
                            name_experiment:str,
                            url_to_mlFLow_server: str = "http://127.0.0.1:5050",
                            url_to_remote_storage: str = "https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow",
                            repo_owner = 'Dimitriy200',
                            repo_name = 'diplom_autoencoder'):
        
        standart_test_params = {
            "Model": "Autoencoder",
            "max_iter": 150,
        }
        
        Train_data = self.get_np_arr_from_csv(path_Train_data)
        Valid_Data = self.get_np_arr_from_csv(path_Valid_Data)
        Predict_data = self.get_np_arr_from_csv(path_Predict_Data)

        new_model = self.create_default_model()


        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_tracking_uri(url_to_remote_storage)    #https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow
        mlflow.set_experiment(name_experiment)
        # mlflow.set_tracking_uri(uri=url_to_mlFLow_server) # https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

        mlflow.keras.autolog()

        with mlflow.start_run():
            
            new_train_model = self.start_train(new_model,
                                            Train_data,
                                            Valid_Data,
                                            params = standart_test_params)
            
            restauriert_data = self.start_predict_model(new_train_model, Predict_data)

            signature = infer_signature(Train_data, restauriert_data)
            
            metrics = self.start_active_validate(restauriert_data,
                                                Predict_data)

            mlflow.log_metric('RMSE', metrics["RMSE"])
            mlflow.log_param('Epochs', '150')

            mlflow.keras.log_model(new_train_model,
                                   artifact_path='my_models',
                                   registered_model_name='autoencoder2')

            # mlflow.sklearn.log_model(
            #     sk_model=new_train_model,
            #     artifact_path="keras-model",
            #     signature=signature,
            #     registered_model_name="keras-module-autoencoder")

        return  metrics, new_train_model



    @classmethod
    def create_default_model(self,
                             input_dim: int = 26) -> tf.keras.Model:

        status_log = ["Create model has successfull", "Create model has error"]

        autoencoder_compressing = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )),
            tf.keras.layers.Dense(16, activation='elu'),

            tf.keras.layers.Dense(10, activation='elu'),
            
            tf.keras.layers.Dense(16, activation='elu'),
            tf.keras.layers.Dense(input_dim, activation='elu')
        ])

        autoencoder_compressing.compile(optimizer = "adam",
                            loss = ["mse"],
                            metrics = ["acc"])

        autoencoder_compressing.summary()
        
        return autoencoder_compressing



    @classmethod
    def start_train(self,
                    model: tf.keras.Model,
                    train_data: np.array,
                    valid_data: np.array,
                    params,
                    epochs = 5,
                    batch_size = 150,) -> tf.keras.Model:
        
        status_log = ["Train successfull", "Train error"]
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        log_dir = "content/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                              histogram_freq=1, 
                                                              profile_batch = (10,100))
        
        # dagshub.init(repo_owner='Dimitriy200', repo_name='diplom_autoencoder', mlflow=True)
        # mlflow.set_tracking_uri("https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow")    #https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow
        # mlflow.set_experiment("New Experiment")

        # mlflow.set_tracking_uri(uri="http://127.0.0.1:5050")

        # # mlflow.start_run()
        # # mlflow.log_params(params)

            
        history = model.fit(train_data, valid_data,
                            shuffle = True,
                            epochs = epochs,
                            batch_size = batch_size,
                            callbacks = [tensorboard_callback],
                            validation_data=(valid_data, valid_data))

            # mlflow.log_metric('RMSE', meric)
            # mlflow.log_param('Epochs', '150')

        
        # mlflow.end_run()

        return model

    

    @classmethod
    def start_predict_model(self, model: tf.keras.Model,
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

        MSE = tf.keras.metrics.MeanSquaredError()
        MSE.update_state(x_x, y_y)
        # RMSE = tf.keras.losses.mean_squared_error(x_x, y_y)
        RMSE = tf.keras.metrics.RootMeanSquaredError()
        RMSE.update_state(x_x, y_y)

        res["MSE"] = MSE.result()
        res["RMSE"] = RMSE.result()

        print(f"\nRMSE = {res['RMSE']}\n")
        
        return res



    @classmethod
    def check_loss(self, inp_DF, res_DF):
        MSE     = tf.keras.losses.mean_squared_error(inp_DF, res_DF)
        RMSE    = tf.keras.metrics.RootMeanSquaredError()
        R2      = tf.keras.metrics.MeanSquaredLogarithmicError()
        MAPE    = tf.keras.metrics.R2Score()

        RMSE.update_state(inp_DF, res_DF)
        R2.update_state(inp_DF, res_DF)
        MAPE.update_state(inp_DF, res_DF)

        print("MSE = ", MSE, "\nRMSE = ", RMSE.result(), "\nR2 = ", R2.result(), "\nMAPE = ", MAPE.result())


    @classmethod
    def start_static_validate(self, 
                              model: tf.keras.models.Model,
                              x_x: np.array,
                              y_y: np.array,):
        
        res = 0

        return res



    @classmethod
    def save_model(self, model: tf.keras.Model,
                   save_filepath: str):
        
        tf.keras.saving.save_model(model,
                                save_filepath)
        
    

    @classmethod
    def load_model(self, load_filepath: str) -> tf.keras.Model:
        new_model = tf.keras.saving.load_model(load_filepath,
                                            custom_objects=None,
                                            compile=True,
                                            safe_mode=True)

        return new_model


    @classmethod
    def save_model_in_MLFlow(self,
                             model: tf.keras.Model,
                             params,
                             metric,
                             X_train,
                             X_test):

        dagshub.init(repo_owner='Dimitriy200', repo_name='diplom_autoencoder', mlflow=True)
        mlflow.set_tracking_uri("https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow")    #https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow

        mlflow.set_tracking_uri(uri="http://127.0.0.1:5050")
        mlflow.set_experiment("New Experiment")

        mlflow.start_run()

        # mlflow.log_param("parameter name ", params)
        # mlflow.log_metric("RMSE", metric)

        mlflow.end_run()

        # with mlflow.start_run():

        #     mlflow.start_run()
        #     # Log the hyperparameters
        #     mlflow.log_params(params)

        #     # Log the loss metric
        #     mlflow.log_metric("accuracy", metric)

        #     # Set a tag that we can use to remind ourselves what this run was for
        #     mlflow.set_tag("Training Info", "Basic Autoencoder model for iris data")

        #     # Infer the model signature
        #     signature = infer_signature(X_train, model.predict(X_train))

        #     # Log the model
        #     model_info = mlflow.sklearn.log_model(
        #         sk_model=model,
        #         artifact_path="autoencoder_model",
        #         signature=signature,
        #         input_example=X_train,
        #         registered_model_name="tracking_model",
        #     )

        #     loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

        #     predictions = loaded_model.predict(X_test)

        #     iris_feature_names = datasets.load_iris().feature_names

        #     subprocess.run("mlflow server --host 127.0.0.1 --port 8080")


            # result = np.array()
            # (X_test, columns=iris_feature_names)
            # result["actual_class"] = y_test
            # result["predicted_class"] = predictions

            # result[:4]


    @classmethod
    def get_np_arr_from_csv(self, path_cfv: str) -> np.array:
        res = genfromtxt(path_cfv, delimiter=',')
        return res