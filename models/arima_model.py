import numpy as np
import pandas as pd
from typing import Dict, Tuple
from statsmodels.tsa.arima.model import ARIMA
from utils.logger import setup_logger
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ArimaPredictor:
    def __init__(self):
        self.logger = setup_logger("ArimaPredictor")
        self.order = (1, 1, 1)
        self.seasonal_order = (1, 1, 1, 12)
        self.model = None

    def train(self, data: Dict[str, np.ndarray]) -> None:
        try:
            returns = data['returns']
            self.model = SARIMAX(
                returns,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False
            )
            self.fitted_model = self.model.fit(
                disp=False,
                maxiter=200,
                method='lbfgs'
            )
            self.logger.info("Modelo ARIMA entrenado exitosamente")
        except Exception as e:
            self.logger.error(f"Error en entrenamiento: {str(e)}")
            raise

    def prepare_data(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        returns = data['returns']
        if len(returns) < 2:
            raise ValueError("Insuficientes datos para ARIMA")
        return returns[~np.isnan(returns)]

    def predict(self, steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if self.fitted_model is None:
                raise ValueError("El modelo debe ser entrenado antes de predecir")
            
            forecast = self.fitted_model.get_forecast(steps=steps)
            mean_forecast = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            return mean_forecast, conf_int
            
        except Exception as e:
            self.logger.error(f"Error en predicción ARIMA: {str(e)}")
            raise

    def evaluate(self, test_data: np.ndarray) -> Dict[str, float]:
        try:
            predictions = self.fitted_model.get_forecast(steps=len(test_data)).predicted_mean
            mse = mean_squared_error(test_data, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_data, predictions)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
            
            self.logger.info("\nMétricas de evaluación:")
            for metric, value in metrics.items():
                self.logger.info(f"{metric.upper()}: {value:.6f}")
                
            return metrics
        except Exception as e:
            self.logger.error(f"Error evaluando modelo: {str(e)}")
            raise 