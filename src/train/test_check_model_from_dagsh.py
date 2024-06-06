import mlflow
import dagshub
import tensorflow as tf


mlflow.set_tracking_uri("https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow")
model_uri = f'models:/autoencoder2/latest'
model = mlflow.keras.load_model(model_uri)


print(model.summary())

