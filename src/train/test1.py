import os
import numpy as np
import logging
import matplotlib.pyplot as plt

from train import Autoencoder_Model

base_dir = os.path.abspath("diplom_autoencoder")
path_searchbarrier = os.path.join(base_dir, "data", "processed", "search_barrier")

logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(os.path.abspath("TrainAutoencoder"),"src", "train", "logs", "train_logs.log" ),
                    filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s")

# --------------------------------------------------------------------


def check_choise_result_barrier():
    tr = Autoencoder_Model()

    tr.choise_result_barrier(path_imp_data = path_searchbarrier)

# --------------------------------------------------------------------


def check_get_mse():
    tr = Autoencoder_Model()
    Control_barrier_Normal = tr.get_np_arr_from_csv(os.path.join(path_searchbarrier, "Control_barrier_Normal.csv"))
    Control_barrier_Anomal = tr.get_np_arr_from_csv(os.path.join(path_searchbarrier, "Control_barrier_Anomal.csv"))

    res = tr.get_mse(Control_barrier_Normal, Control_barrier_Anomal)

    # print(f"Control_barrier_Normal = {Control_barrier_Normal}")
    # print(f"\nControl_barrier_Anomal = {Control_barrier_Anomal}")
    print(f"res = {res}")

# check_get_mse()


# --------------------------------------------------------------------

def fit_model_and_save_dagshub():
    tr = Autoencoder_Model()

    new_model = tr.create_default_model()

    train_data = tr.get_np_arr_from_csv(os.path.join(base_dir, "data", "final", "train_and_test", "train.csv"))
    test_data = tr.get_np_arr_from_csv(os.path.join(base_dir, "data", "final", "train_and_test", "test.csv"))

    epochs = 10
    new_trained_model = tr.start_train(new_model,
                                       train_data = train_data,
                                       valid_data = test_data,
                                       epochs = epochs)
    
    predict_data = tr.start_predict_model(model = new_trained_model,
                                          predict_data = test_data)
    
    rmse = tr.get_rmse(x_x_true = test_data,
                       y_y_pred = predict_data)
    
    tr.save_model_in_MLFlow(saved_model = new_trained_model,
                            mlfl_tracking_username = "a1482d904ec14cd6e61aa6fcc9df96278dc7c911",
                            metrics = rmse,
                            metrics_name = "MSE",
                            epochs = epochs)

# fit_model_and_save_dagshub()


# --------------------------------------------------------------------
def check_start_train_and_save_mlflow():
    tr = Autoencoder_Model()
    train_data = os.path.join(base_dir, "data", "final", "train_and_test", "train.csv")
    test_data = os.path.join(base_dir, "data", "final", "train_and_test", "test.csv")
    predict_data = os.path.join(base_dir, "data", "final", "static_valid", "Satic_validation_Normal.csv")

    all_normal =  os.path.join(base_dir, "data", "processed", "normal", "Normal.csv")

    new_model = tr.start_train_and_save_mlflow(path_Train_data = train_data,
                                                path_Valid_Data = test_data,
                                                path_Predict_Data = predict_data,
                                                name_experiment = "MAIN_EXPERIMENT",
                                                registered_model_name = "Barrier_test_model",
                                                mlfl_tr_username = "a1482d904ec14cd6e61aa6fcc9df96278dc7c911",
                                                epochs = 80,
                                                batch_size = 200)


check_start_train_and_save_mlflow()


# --------------------------------------------------------------------
def check_choise_result_barrier():
    tr = Autoencoder_Model()

    path_imp_data = os.path.join(base_dir, "data", "processed", "search_barrier")

    model = tr.load_model_from_MlFlow(dagshub_toc_username = "Dimitriy200",
                                      dagshub_toc_pass = "%",
                                      dagshub_toc_tocen = "",
                                      name_model = "Barrier_test_model")

    mse_Normal, mse_anomal = tr.choise_result_barrier(path_choise_Normal = os.path.join(path_imp_data, "Choise_barrier_Normal.csv"),
                                                      path_control_Normal = os.path.join(path_imp_data, "Control_barrier_Normal.csv"),
                                                      path_choise_Anomal = os.path.join(path_imp_data, "Choise_barrier_Anomal.csv"),
                                                      path_control_Anomal = os.path.join(path_imp_data, "Control_barrier_Anomal.csv"),
                                                      model = model)
    
    all_mse = np.concatenate([mse_Normal, mse_anomal], axis=0)
    logging.info(f"all_mse shape = {all_mse.shape}")

    fig, ax = plt.subplots()
    ax.scatter(all_mse[:, 0], all_mse[:, 1], vmin=0)
    
    plt.style.use('_mpl-gallery')
    plt.show()

check_choise_result_barrier()


# --------------------------------------------------------------------