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
        
        try:
            model = tf.keras.Sequential([
                # Entrada
                tf.keras.layers.LSTM(64, input_shape=input_shape,
                                   return_sequences=True),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                
                # Segunda capa
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                
                # Capa densa adicional
                tf.keras.layers.Dense(32, activation='relu'),
                
                # Salida para regresión
                tf.keras.layers.Dense(1, activation='linear')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            model.summary()
            return model
        except Exception as e:
            self.logger.error(f"Error construyendo modelo LSTM: {str(e)}")
            raise

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 150):
        self.logger.info("Entrenando modelo LSTM...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                min_delta=1e-4
            )
        ]
        
        self.model.fit(
            X_train, 
            y_train,
            epochs=epochs,
            batch_size=32,  # Batch size más pequeño
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        self.logger.info("Modelo LSTM entrenado exitosamente")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self.logger.info("Realizando predicciones con LSTM...")
        return self.model.predict(X_test)