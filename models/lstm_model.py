import tensorflow as tf
import numpy as np
from typing import Tuple
from utils.logger import setup_logger

class LSTMPredictor:
    def __init__(self, input_shape: Tuple[int, int]):
        self.logger = setup_logger("LSTMPredictor")
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Sequential:
        self.logger.info(f"Construyendo modelo LSTM con input_shape: {input_shape}")
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        model.summary()
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, batch_size: int = 32):
        self.logger.info("Entrenando modelo LSTM...")
        self.model.fit(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_split=0.1,
            shuffle=True
        )
        self.logger.info("Modelo LSTM entrenado exitosamente")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.logger.info("Realizando predicciones con LSTM...")
        predictions = self.model.predict(X_test)
        return predictions