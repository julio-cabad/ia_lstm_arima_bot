import numpy as np
from typing import Dict, Tuple
from utils.logger import setup_logger
from models.lstm_model import LSTMPredictor
from models.arima_model import ArimaPredictor

class EnsemblePredictor:
    def __init__(self):
        self.logger = setup_logger("EnsemblePredictor")
        self.lstm = None  # Lo inicializaremos en train cuando tengamos el input_shape
        self.arima = ArimaPredictor()
        self.weights = {
            'lstm': 0.5,
            'arima': 0.5
        }
        self.timeframe = "1h"
    
    def train(self, data: Dict[str, np.ndarray]) -> None:
        try:
            # Inicializar LSTM con el input_shape correcto
            X = data['lstm']['X']
            input_shape = (X.shape[1], X.shape[2])
            self.lstm = LSTMPredictor(input_shape)
            
            # Entrenar ambos modelos
            self.lstm.train(data['lstm']['X'], data['lstm']['y'])
            self.arima.train(data['arima'])
            
            self.logger.info("Ensemble entrenado exitosamente")
        except Exception as e:
            self.logger.error(f"Error en entrenamiento ensemble: {str(e)}")
            raise
    
    def predict(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        try:
            # Obtener predicciones de ambos modelos
            lstm_pred = self.lstm.predict(data['lstm']['X'])
            arima_pred = self.arima.predict(data['arima'])[0]
            
            # Combinar predicciones
            ensemble_pred = (
                lstm_pred * self.weights['lstm'] + 
                arima_pred * self.weights['arima']
            )
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"Error en predicción ensemble: {str(e)}")
            raise 