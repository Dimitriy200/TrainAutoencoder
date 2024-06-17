import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
                                                mlfl_tr_username = "",
                                                epochs = 50,
                                                batch_size = 100)

# check_start_train_and_save_mlflow()


# --------------------------------------------------------------------

def scatter_subplots(all_mse: np.array):

    str_mse, col = all_mse.shape
    indexses = np.array(range(1, str_mse + 1))
    fig, ax = plt.subplots()

    logging.info(f"all_mse = {all_mse}, \nindexses = {indexses}")
    logging.info(f"all_mse = {all_mse.shape} \nindexses = {indexses.shape}")

    big_all_mse, small_all_mse = train_test_split(all_mse,
                                                  random_state=0,
                                                  train_size = .98)

    str_small_mse, col = small_all_mse.shape
    indexses_small = np.array(range(1, str_small_mse + 1))
    logging.info(f"big_all_mse = {big_all_mse.shape} \nsmall_all_mse = {small_all_mse.shape}")

    ax.scatter(all_mse[:, 0],
               all_mse[:, 1],
               vmin=0)

    plt.title("class and mse -> Norm and Anom")
    plt.xlabel("mse")
    plt.ylabel("class")

    # ax.scatter(all_mse[:, 0],
    #            indexses,
    #            vmin=0)
    
    # plt.title("index and mse -> Normal")
    # plt.xlabel("mse")
    # plt.ylabel("index") 
    
    plt.style.use('_mpl-gallery')
    plt.show()

# ----------------------------


def check_choise_result_barrier():
    tr = Autoencoder_Model()

    path_imp_data = os.path.join(base_dir, "data", "processed", "search_barrier")
    path_static_val_data = os.path.join(base_dir, "data", "final", "static_valid")

    model = tr.load_model_from_MlFlow(dagshub_toc_username = "Dimitriy200",
                                      dagshub_toc_pass = "RamZaZ3961%",
                                      dagshub_toc_tocen = "a1482d904ec14cd6e61aa6fcc9df96278dc7c911",
                                      name_model = "Barrier_test_model")

    mse_Normal, mse_Anomal, mse_met_anom = tr.choise_result_barrier(path_choise_Normal = os.path.join(path_imp_data, "Choise_barrier_Normal.csv"),
                                                      path_control_Normal = os.path.join(path_imp_data, "Control_barrier_Normal.csv"),
                                                      path_choise_Anomal = os.path.join(path_imp_data, "Choise_barrier_Anomal.csv"),        #Satic_validation_Anomal.csv
                                                      path_control_Anomal = os.path.join(path_imp_data, "Control_barrier_Anomal.csv"),      #Satic_validation_Normal.csv
                                                      model = model)
    
    all_mse = np.concatenate([mse_Normal, mse_Anomal], axis=0)
    logging.info(f"all_mse shape = {all_mse.shape}")

    scatter_subplots(all_mse)
    # scatter_subplots(mse_anomal)

    np.savetxt(os.path.join(base_dir, "data", "raw", "tests", "mse_met_anom.csv"), mse_met_anom, delimiter=',')

check_choise_result_barrier()


# --------------------------------------------------------------------
def one_obj_in_model():
    tr = Autoencoder_Model()
    
    norm_obj = np.array([[-1.795721061982818778e+00,-1.129204285497100946e+00,-1.041162806900336690e+00,-1.115689788543209549e+00,3.459547318756359680e-01,1.079184532391066487e+00,1.062163245397369282e+00,9.262910700811647358e-01,9.674534588243941524e-01,1.107714212952182642e+00,1.115017637154611663e+00,1.119257725252984503e+00,8.013997263425985951e-01,1.043826886626678219e+00,1.029504646746983942e+00,8.049281276177563393e-01,1.120572830328817604e+00,3.449296373543788707e-01,7.283160321483066468e-01,-9.064211994673586625e-01,9.635893449131582855e-01,9.445499589932081497e-01,8.016547624035603725e-01,3.459547318756358014e-01,1.117685454880845031e+00,1.118210856750607940e+00]])
    anom_obj = np.array([[-7.043365496037183870e-01,-1.593693459961881623e-02,1.499825848743208345e+00,1.173832006804681027e+00,3.459547318756359680e-01,-1.342554116894565164e+00,-1.126714748545547984e+00,-9.273175614629300956e-01,-1.019418880166842234e+00,-1.403104354932863229e+00,-1.352431324614069119e+00,-1.271784189064844250e+00,-4.351005669331757808e-01,-9.684441716987520765e-01,-9.409039885264448566e-01,-6.806257369195612972e-01,-1.264730289684065756e+00,3.441200438198444012e-01,-1.375658517944799986e-01,3.573187400873057418e-01,-1.037786485787804969e+00,-1.054074639582845307e+00,-4.333231520025237482e-01,3.459547318756358014e-01,-1.313164177857177428e+00,-1.300647438942920564e+00]])
    logging.info(f"norm_obj = {norm_obj}")
    logging.info(f"anom_obj = {anom_obj}")

    model = tr.load_model_from_MlFlow(dagshub_toc_username = "Dimitriy200",
                                      dagshub_toc_pass = "RamZaZ3961%",
                                      dagshub_toc_tocen = "a1482d904ec14cd6e61aa6fcc9df96278dc7c911",
                                      name_model = "Barrier_test_model")
    
    pred_Norm_obj = tr.start_predict_model(model, predict_data = norm_obj)
    pred_Anom_obj = tr.start_predict_model(model, predict_data = anom_obj)
    logging.info(f"pred_Norm_obj = {pred_Norm_obj}")
    logging.info(f"pred_Anom_obj = {pred_Anom_obj}")

    mse_norm = tr.get_mse(norm_obj, pred_Norm_obj)
    mse_anom = tr.get_mse(anom_obj, pred_Anom_obj)
    logging.info(f"mse_norm = {mse_norm}")
    logging.info(f"mse_anom = {mse_anom}")

# one_obj_in_model()
# --------------------------------------------------------------------