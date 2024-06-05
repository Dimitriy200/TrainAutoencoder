import os
import numpy as np

from train import Autoencoder_Model

base_dir = os.path.abspath("diplom_autoencoder")

train_data  = os.path.join(base_dir, "data", "final", "Train.csv")
predict_data  = os.path.join(base_dir, "data", "final", "Predict.csv")

print(f"\n base_dir - {base_dir}")
print(train_data)
 

mod = Autoencoder_Model()
res = mod.start_all_processes(train_data,
                        train_data,
                        predict_data)

print(res)