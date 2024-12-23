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
        # Configuración óptima basada en mejores resultados
        self.order = (1, 1, 1)
        self.seasonal_order = (1, 1, 1, 12)
        self.enforce_invertibility = True
        self.concentrate_scale = True
        
        # Parámetros de optimización ajustados
        self.optimization_params = {
            'method': 'lbfgs',
            'maxiter': 300,
            'factr': 1e7,  # Para evitar warnings
            'maxfun': 300,
            'epsilon': 1e-8
        }
        self.model = None

    def train(self, data: Dict[str, np.ndarray]) -> None:
        try:
            returns = data['returns']
            self.model = SARIMAX(
                returns,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_invertibility=self.enforce_invertibility,
                concentrate_scale=self.concentrate_scale
            )
            self.fitted_model = self.model.fit(
                disp=False,
                **self.optimization_params
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

    def grid_search_parameters(self, data):
        try:
            # Expandir búsqueda alrededor de los parámetros exitosos
            orders = [
                (1,1,1),  # Parámetro base exitoso
                (2,1,1),  # Aumentar AR
                (1,1,2),  # Aumentar MA
                (2,1,2),  # Aumentar ambos
                (1,2,1)   # Más diferenciación
            ]
            
            seasonal_orders = [
                (1,1,1,12),  # Parámetro base exitoso
                (1,1,1,24),  # Ciclo diario
                (2,1,1,12),  # Más AR estacional
                (1,1,2,12)   # Más MA estacional
            ]
            
            # Modificar parámetros de optimización para suprimir salida
            optimization_params = self.optimization_params.copy()
            optimization_params.update({
                'disp': False,
                'warning_only': True,
                'print_level': 0,
                'maxiter': 300,  # Reducir iteraciones
                'tol': 1e-3,    # Tolerancia más flexible
                'optim_complex_step': False  # Desactivar para velocidad
            })
            
            best_aic = float('inf')
            best_params = None
            best_metrics = None
            
            print("\nBuscando mejores parámetros...")
            total_combinations = len(orders) * len(seasonal_orders)
            current = 0
            
            for order in orders:
                for seasonal_order in seasonal_orders:
                    current += 1
                    print(f"Progreso: {current}/{total_combinations}", end='\r')
                    
                    try:
                        model = SARIMAX(
                            data, 
                            order=order, 
                            seasonal_order=seasonal_order,
                            enforce_invertibility=self.enforce_invertibility,
                            concentrate_scale=self.concentrate_scale
                        )
                        fitted = model.fit(**optimization_params)
                        
                        predictions = fitted.get_forecast(steps=50).predicted_mean
                        win_rate = np.mean(np.sign(predictions[1:]) == np.sign(data[1:50]))
                        score = fitted.aic - (win_rate * 1000)
                        
                        if score < best_aic:
                            best_aic = score
                            best_params = (order, seasonal_order)
                            best_metrics = {'win_rate': win_rate, 'aic': fitted.aic}
                            
                    except:
                        continue
            
            print("\nMejores métricas encontradas:")
            print(f"Win Rate: {best_metrics['win_rate']:.2%}")
            print(f"AIC: {best_metrics['aic']:.2f}")
            print(f"Mejores parámetros: {best_params}")
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error en grid search: {str(e)}")
            raise