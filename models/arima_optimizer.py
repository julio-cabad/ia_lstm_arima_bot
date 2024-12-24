import numpy as np
import pandas as pd
from typing import Dict, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.logger import setup_logger
import json

class ArimaOptimizer:
    def __init__(self):
        self.logger = setup_logger("ArimaOptimizer")
        
        # Parámetros de búsqueda
        self.orders = []
        
        self.seasonal_orders = [
            (1,1,1,12),  # Base
            (1,1,1,24),  # Ciclo diario
            (2,1,1,12),  # Más AR estacional
            (1,1,2,12)   # Más MA estacional
        ]
        
        self.optimization_params = {
            'method': 'lbfgs',
            'maxiter': 300,
            'disp': False
        }
    
    def optimize(self, data: Dict[str, np.ndarray]) -> Dict:
        """Encuentra los mejores parámetros ARIMA"""
        try:
            best_metrics = None
            best_params = None
            best_sharpe = -float('inf')
            
            total = len(self.orders) * len(self.seasonal_orders)
            print(f"\nProbando {total} combinaciones de parámetros...")
            
            for order in self.orders:
                for seasonal_order in self.seasonal_orders:
                    try:
                        model = SARIMAX(
                            data['returns'],
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_invertibility=True,
                            concentrate_scale=True
                        )
                        fitted = model.fit(**self.optimization_params)
                        
                        # Evaluar modelo
                        predictions = fitted.get_forecast(steps=50).predicted_mean
                        metrics = self._evaluate_predictions(predictions, data['returns'][-50:])
                        
                        if metrics['sharpe'] > best_sharpe:
                            best_sharpe = metrics['sharpe']
                            best_metrics = metrics
                            best_params = {
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'enforce_invertibility': True,
                                'concentrate_scale': True
                            }
                            
                    except:
                        continue
            
            # Guardar mejores parámetros
            with open('best_arima_params.json', 'w') as f:
                json.dump(best_params, f)
                
            return best_params, best_metrics
            
        except Exception as e:
            self.logger.error(f"Error en optimización: {str(e)}")
            raise 