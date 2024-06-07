import os
import numpy as np

from train import Autoencoder_Model

base_dir = os.path.abspath("diplom_autoencoder")

train_data  = os.path.join(base_dir, "data", "final", "Train.csv")
predict_data  = os.path.join(base_dir, "data", "final", "Predict.csv")

print(f"\n base_dir - {base_dir}")
print(train_data)
 

mod = Autoencoder_Model()
metrics, model = mod.start_all_processes(train_data,
                        train_data,
                        predict_data,
                        name_experiment = "exp 3")

res = metrics["RMSE"]

print(f"RMSE = {res}")


#
# classifiers = [
#     "Programming Language :: Python :: 3",
#     "License :: OSI Approved :: MIT License",
#     "Operating System :: OS Independent",
# ]

# "scipy>=1.6.0",
# "joblib>=1.2.0",
# "threadpoolctl>=3.1.0",
# "numpy>=1.19.5",