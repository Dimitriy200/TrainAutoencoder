import dagshub.auth
import numpy as np
import keras
# import tensorflow as tf
import datetime
import mlflow
import subprocess
import dagshub
import os
import logging


# from tensorflow import keras
# from tensorflow.python import keras#
from mlflow.models import infer_signature
from numpy import genfromtxt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from mlflow.models import infer_signature
from mlflow.environment_variables import MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_TOKEN, MLFLOW_TRACKING_AUTH

# from tensorflow import keras

# mlflow server --host 127.0.0.1 --port 5050



class Autoencoder_Model():

    def __init__(self) -> None:
        pass
        #%load_ext tensorboard
        # logging.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        

        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir = log_dir, 
        #                                                    histogram_freq = 1, 
        #                                                    profile_batch = (10, 100))


    @classmethod
    def start_train_and_save_mlflow(self,
                                    path_Train_data: str,
                                    path_Valid_Data: str,
                                    path_Predict_Data: str, #Должно быть 2 строки данныъ. Первую можно заполнить нулями
                                    name_experiment: str,
                                    mlfl_tr_username,
                                    url_to_remote_storage: str = "https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow",
                                    repo_owner = 'Dimitriy200',
                                    repo_name = 'diplom_autoencoder',
                                    registered_model_name = "autoencoder_3",
                                    epochs = 5,
                                    batch_size = 80):
        
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlfl_tr_username

        standart_test_params = {
            "Model": "Autoencoder",
            "max_iter": 150,
        }

        train_data = self.get_np_arr_from_csv(path_Train_data)
        valid_Data = self.get_np_arr_from_csv(path_Valid_Data)
        predict_data = self.get_np_arr_from_csv(path_Predict_Data)

        # new_model = self.create_default_model()
        new_model = self.create_sparse_autoencoders()

        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_tracking_uri(url_to_remote_storage)    #https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow
        mlflow.set_experiment(name_experiment)
        
        mlflow.keras.autolog()

        with mlflow.start_run():
            
            new_train_model = self.start_train(model = new_model,
                                               train_data = train_data,
                                               valid_data = valid_Data,
                                               params = standart_test_params,
                                               epochs = epochs,
                                               batch_size = batch_size)
            
            restauriert_data = self.start_predict_model(new_train_model, predict_data)

            signature = infer_signature(train_data, restauriert_data)
            
            rmse = self.get_rmse(x_x_true = predict_data,
                                y_y_pred = restauriert_data)

            mlflow.log_metric('rmse', rmse)
            mlflow.log_param('Epochs', epochs)

            mlflow.keras.log_model(new_train_model,
                                   artifact_path = 'my_models',
                                   registered_model_name = registered_model_name)

            # mlflow.sklearn.log_model(
            #     sk_model=new_train_model,
            #     artifact_path="keras-model",
            #     signature=signature,
            #     registered_model_name="keras-module-autoencoder")

        return new_train_model


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

    # СОЗДАЕМ МОДЕЛЬ СТАНДАРТНОГО [Undercomplete Autoencoders]
    @classmethod
    def create_default_model(self,
                             input_dim: int = 26) -> keras.Model:

        status_log = ["Create model has successfull", "Create model has error"]
        mae = keras.metrics.MeanAbsoluteError()
        rmse = keras.metrics.RootMeanSquaredError(name = "rmse")

        autoencoder_compressing = keras.models.Sequential([
            keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )),
            keras.layers.Dense(16, activation='elu'),
            
            keras.layers.Dense(10, activation='elu'),
            
            keras.layers.Dense(16, activation='elu'),
            keras.layers.Dense(input_dim, activation='elu')
        ])

        autoencoder_compressing.compile(optimizer = "adam",
                            loss = ["mse"],
                            metrics = [mae, rmse])

        autoencoder_compressing.summary()
        logging.info(autoencoder_compressing.summary())
        
        return autoencoder_compressing


    # СОЗДАЕМ МОДЕЛЬ [Sparse Autoencoders]
    @classmethod
    def create_sparse_autoencoders(self,
                             input_dim: int = 26) -> keras.Model:

        status_log = ["Create model has successfull", "Create model has error"]
        mae = keras.metrics.MeanAbsoluteError()
        rmse = keras.metrics.RootMeanSquaredError(name = "rmse")

        autoencoder_compressing = keras.models.Sequential([
            keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )),
            keras.layers.Dense(30, activation='elu'),
            
            keras.layers.Dense(36, activation='elu'),
            
            keras.layers.Dense(30, activation='elu'),
            keras.layers.Dense(input_dim, activation='elu')
        ])

        autoencoder_compressing.compile(optimizer = "adam",
                            loss = ["mse"],
                            metrics = [mae, rmse])

        autoencoder_compressing.summary()
        logging.info(autoencoder_compressing.summary())
        
        return autoencoder_compressing

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------


    @classmethod
    def add_train(self,
                  path_to_train,
                  path_to_valid,
                  dagshub_toc_username,
                  dagshub_toc_pass,
                  dagshub_toc_tocen,
                  epochs = 5,
                  batch_size = 50,
                  name_model: str = "Barrier_test_model"):
        
        train_data = self.get_np_arr_from_csv(path_to_train)
        valid_data = self.get_np_arr_from_csv(path_to_valid)
        logging.info(f"train_data = {train_data.shape}")
        logging.info(f"valid_data = {valid_data.shape}")
        
        model = self.load_model_from_MlFlow(dagshub_toc_username=dagshub_toc_username,
                                            dagshub_toc_pass=dagshub_toc_pass,
                                            dagshub_toc_tocen=dagshub_toc_tocen)
        
        new_model = self.start_train(model=model,
                                     train_data = train_data,
                                     valid_data = valid_data)
        
        return new_model
        

    @classmethod
    def start_train(self,
                    model: keras.Model,
                    train_data: np.array,
                    valid_data: np.array,
                    params,
                    epochs = 20,
                    batch_size = 80,) -> keras.Model:
        
        status_log = ["Train successfull", "Train error"]

        log_dir = "content/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        logging.info(f"START TRAIN, PARAMETRES: EPOCHS = {epochs}\nBATCH_SIZE = {batch_size}, \n TRAIN_DATA_SIZE = {train_data.shape}")

        history = model.fit(train_data, train_data,
                            shuffle = True,
                            epochs = epochs,
                            batch_size = batch_size,
                            # callbacks = [tensorboard_callback],
                            validation_data=(valid_data, valid_data))

        return model


    @classmethod
    def start_predict_model(self,
                            model: keras.Model,
                            predict_data: np.array,
                            batch_size: int = 200) -> np.array:

        logging.info("START PREDICT")
        res_data = model.predict(predict_data)

        return res_data
        

    @classmethod
    def get_class_from_object(self,
                            model: keras.Model,
                            input_data: np.array,
                            batch_size: int = 200,
                            barrier_line = .1) -> np.array:
        
        logging.info("START GET CLASS OBJECT")
        predicted_data = model.predict(input_data)

        arr_mse = self.get_mse(input_data, predicted_data)

        valid_arr = []
        for mse in arr_mse:
            if mse < barrier_line:
                valid_arr.append(1)
            else:
                valid_arr.append(0)
        
        np_valid = np.array(valid_arr)
        res_data = np.rot90(np.stack([arr_mse, np_valid]), k=1)
        
        logging.info(f"res_data = {res_data}")

        return np_valid


    @classmethod
    def get_mse(self,
                x_x_true: np.array,
                y_y_pred: np.array) -> np.array :

        # MSE = keras.metrics.MeanSquaredError()
        MSE = keras.losses.mean_squared_error(x_x_true, y_y_pred)
        # MSE.update_state(x_x, y_y)
        
        # res_mse = MSE.result()
        res_mse_np = np.array(MSE)
        
        return res_mse_np


    @classmethod
    def get_rmse(self,
                x_x_true: np.array,
                y_y_pred: np.array) :
        
        RMSE = keras.metrics.RootMeanSquaredError()
        RMSE.update_state(x_x_true, y_y_pred)

        res_rmse = RMSE.result()
        logging.info(f"RMSE = {res_rmse}")


        np_rmse = np.array(res_rmse)
        logging.info(f"NP_RMSE = {np_rmse}")

        return np_rmse


    @classmethod
    def get_bin_acc(self,
                    x_x_true: np.array,
                    y_y_pred: np.array) -> np.array:
        
        ACC = keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)
        ACC.update_state(x_x_true, y_y_pred)

        res = ACC.result()

        return res


    @classmethod
    def get_confusuon_matrix_for_class(self,
                                        x_x_true: np.array,
                                        y_y_pred: np.array) -> np.array:
        
        tn, fp, fn, tp = confusion_matrix(x_x_true, y_y_pred).ravel()

        return tn, fp, fn, tp


    @classmethod
    def get_metrics(self, x_x, y_y):
        R2      = keras.metrics.MeanSquaredLogarithmicError()
        MAPE    = keras.metrics.R2Score()

        R2.update_state(x_x, y_y)
        MAPE.update_state(x_x, y_y)

        res_r2 = R2.result()
        res_MAPE = MAPE.result()

        res_R2_np = np.array(res_r2)
        res_MAPE_np = np.array(res_MAPE)

        return res_R2_np, res_MAPE_np


    @classmethod
    def start_static_validate(self, 
                              model: keras.models.Model,
                              x_x: np.array,
                              y_y: np.array,):
        
        res = 0

        return res


    @classmethod
    def save_model_in_MLFlow(self,
                             saved_model: keras.Model,
                             mlfl_tracking_username: str,
                             metrics,
                             metrics_name: str,
                             epochs: str,
                             url_to_remote_storage: str = "https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow",
                             repo_owner: str = 'Dimitriy200',
                             repo_name: str = 'diplom_autoencoder'):
        
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlfl_tracking_username

        dagshub.init(repo_owner=repo_owner,
                     repo_name=repo_name,
                     mlflow=True)
        
        mlflow.set_tracking_uri(url_to_remote_storage)
        mlflow.set_experiment("New_Experiment_mean_model")

        mlflow.start_run()
        mlflow.keras.autolog()

        with mlflow.start_run(nested=True):

            mlflow.log_metric(metrics_name, metrics)
            mlflow.log_param('Epochs', epochs)

            mlflow.keras.log_model(saved_model,
                                   artifact_path='my_models',
                                   registered_model_name='autoencoder_main')

        mlflow.end_run()


    @classmethod
    def load_model_from_MlFlow(self,
                               dagshub_toc_username,
                               dagshub_toc_pass,
                               dagshub_toc_tocen,
                               uri: str = "https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow",
                               name_model: str = "Barrier_test_model",
                               version_model = "latest",):
        
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_toc_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_toc_pass
        os.environ['MLFLOW_TRACKING_TOKEN'] = dagshub_toc_tocen

        dagshub.auth.add_app_token(token=dagshub_toc_tocen)

        dagshub.init(repo_owner='Dimitriy200', repo_name='diplom_autoencoder', mlflow=True)
        mlflow.set_tracking_uri(uri)

        model_uri = f'models:/{name_model}/{version_model}'

        with mlflow.start_run():
            model = mlflow.keras.load_model(model_uri)

        print(model.summary())

        return model


    @classmethod
    def choise_barrier_line(self,
                    path_choise_Normal,
                    path_control_Normal,
                    path_choise_Anomal,
                    path_control_Anomal,
                #   path_out_data: str,
                    model: keras.Model):
        
        barrier_line = 0

        choise_Normal = self.get_np_arr_from_csv(path_choise_Normal)
        control_Normal = self.get_np_arr_from_csv(path_control_Normal)
        choise_Anomal = self.get_np_arr_from_csv(path_choise_Anomal)
        control_Anomal = self.get_np_arr_from_csv(path_control_Anomal)

        logging.info(f"INPUR DATA: \nchoise_Normal = {choise_Normal.shape}\ncontrol_Normal = {control_Normal.shape}\nchoise_Anomal = {choise_Anomal.shape}\ncontrol_Anomal = {control_Anomal.shape}")

        # 8. цикл для каждого положения разделяющей поверхности от mse min до mse max делать:
        #   { для кадого объекта определяем считать его нормальным или аномальным в соответствии с положением разд поверхности
        #     для каждого объекта определяем: к какому классу ошибок он относится. TP FP TN FN.
        #     расчитываем метрику для датасета.
        #     запоминаем знач метрики соответсв данному положению разд поверхн}

        # Прогоняем данные через обученную модель
        predicted_control_Normal = self.start_predict_model(model = model,
                                                            predict_data = control_Normal)
        
        predicted_control_Anom = self.start_predict_model(model = model,
                                                       predict_data = control_Anomal)

        # Получаем массив ошибок mse для каждого датасета
        mse_Normal = self.get_mse(control_Normal, predicted_control_Normal)
        mse_Anomal = self.get_mse(control_Anomal, predicted_control_Anom)

        np_mse_Normal = np.array(mse_Normal)
        np_mse_Anomal = np.array(mse_Anomal)
       
        logging.info(f"METRICS: \nnp_mse_Normal = {np_mse_Normal}   \nnp_mse_Anomal = {np_mse_Anomal}   \nshapes = {np_mse_Normal.shape}")
        
        str_df = np_mse_Normal.shape
        
        # Нормальные значения = класс 1 
        # Аномальные значения = класс 0 
        metrics_Norm = np.ones(shape = (str_df))
        metrics_Anom = np.zeros(shape = (str_df))
        logging.info(f"metrics_Norm = {metrics_Norm.shape}")

        metrics_mse_Normal = np.stack([np_mse_Normal, metrics_Norm])
        metrics_mse_Anomal = np.stack([np_mse_Anomal, metrics_Anom])
        logging.info(f"METRICS: \nmetrics_mse_Normal = \n{metrics_mse_Normal}\nmetrics_mse_Anomal = \n{metrics_mse_Anomal}\nshapes = {metrics_mse_Normal.shape}")

        rsh_metrics_mse_Normal = np.rot90(metrics_mse_Normal, k= 1)
        rsh_metrics_mse_Anomal = np.rot90(metrics_mse_Anomal, k= 1)
        logging.info(f"RSHPE METRICS: \nmetrics_mse_Normal = \n{rsh_metrics_mse_Normal}\nmetrics_mse_Anomal = \n{rsh_metrics_mse_Anomal}")

        all_mse = np.concatenate([metrics_mse_Normal, metrics_mse_Anomal])

        # barrier_line = np.array((max(rsh_metrics_mse_Anomal[:, 0]) + min(rsh_metrics_mse_Normal[:, 0])) / 2)
        
        logging.info(f"barrier_line = {barrier_line}")
        
        all_mse = np.concatenate([rsh_metrics_mse_Normal, rsh_metrics_mse_Anomal], axis=0)

        
        # Подбор разделяющей поверхности
        list_mse = all_mse[:, 0]
        sort_list_mse = np.sort(list_mse, kind = 'mergesort')
        logging.info(f"list_mse = {list_mse}, shapes = {list_mse.shape}")

        barrier_store = []
        acc_arr = []
        for mse in sort_list_mse:
            barrier_line = mse
            valid_arr = []

            for mse in list_mse:
                if mse < barrier_line:
                    valid_arr.append(1)
                else:
                    valid_arr.append(0)
            
            acc_m = self.get_bin_acc(all_mse[:, 1], valid_arr)
            acc_arr.append(acc_m)
            barrier_store.append(barrier_line)

        # logging.info(f"acc_arr = {acc_arr}")
        # logging.info(f"barrier_store = {barrier_store}")

        max_acc = np.max(acc_arr)
        logging.info(f"max_acc = {max_acc}")
        index = 0
        for i in acc_arr:
            if i == max_acc:
                break
            
            index += 1

        res_barrier_line = barrier_store[index]
        logging.info(f"index_max_acc = {index}")
        logging.info(f"res_barrier_line = {res_barrier_line}")
        
        # Получим результат
        valid_arr = []
        for mse in list_mse:
            if mse < res_barrier_line:
                valid_arr.append(1)
            else:
                valid_arr.append(0)
        
        # Собираем все данные в единый фрейм    =>    [mse,   True_Class,   Predict_Class]
        np_valid_arr = np.array(valid_arr)
        d_valid_arr = np.atleast_2d(np_valid_arr)
        rsh_valid_arr = np.rot90(d_valid_arr, k= -1)
        logging.info(f"rsh_valid_arr = {rsh_valid_arr}")

        mse_met_ALL = np.column_stack((all_mse, rsh_valid_arr))
        logging.info(f"np_valid_arr = {mse_met_ALL}")


        return res_barrier_line, mse_met_ALL


    @classmethod
    def save_model(self, model: keras.Model,
                   save_filepath: str):
        
        keras.saving.save_model(model,
                                save_filepath)
        

    @classmethod
    def load_model(self, load_filepath: str) -> keras.Model:
        new_model = keras.saving.load_model(load_filepath,
                                            custom_objects=None,
                                            compile=True,
                                            safe_mode=True)

        return new_model


    @classmethod
    def get_np_arr_from_csv(self, path_cfv: str) -> np.array:
        res = genfromtxt(path_cfv, delimiter=',')
        return res