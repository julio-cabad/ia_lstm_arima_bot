import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback
from sklearn.model_selection import TimeSeriesSplit
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collector import DataCollector
from data.preprocessor import DataPreprocessor
from models.arima_model import ArimaPredictor
from models.lstm_model import LSTMPredictor
from utils.logger import setup_logger

class TestArima:
    def __init__(self):
        self.logger = setup_logger("TestArima")
        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        
    def evaluate_predictions(self, results: pd.DataFrame, silent: bool = False) -> dict:
        """Evaluación detallada de predicciones"""
        try:
            # Métricas básicas
            mse = mean_squared_error(results['Real'], results['Predicción'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(results['Real'], results['Predicción'])
            
            # Métricas direccionales mejoradas
            correct_direction = np.mean(
                (results['Real'] * results['Predicción']) > 0
            )
            
            # Métricas de trading
            results['Signal'] = np.sign(results['Predicción'])
            results['Return'] = results['Real'] * results['Signal'].shift(1)
            
            win_rate = len(results[results['Return'] > 0]) / len(results[results['Return'] != 0])
            sharpe = np.sqrt(252) * results['Return'].mean() / results['Return'].std()
            
            if not silent:
                print("\n=== Evaluación Detallada ===")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"Dirección Correcta: {correct_direction:.2%}")
                print(f"Win Rate: {win_rate:.2%}")
                print(f"Sharpe Ratio: {sharpe:.4f}")
            
            return {
                'rmse': rmse,
                'mae': mae,
                'correct_direction': correct_direction,
                'win_rate': win_rate,
                'sharpe': sharpe
            }
            
        except Exception as e:
            self.logger.error(f"Error en evaluación: {str(e)}")
            raise

    def test_arima_prediction(self):
        """Prueba el modelo ARIMA"""
        try:
            print("\nIniciando prueba de ARIMA...")
            
            # Obtener datos
            symbol = "BTCUSDT"
            timeframe = "1h"
            df = self.collector.get_historical_data(symbol, timeframe)
            df = df.tail(2000)
            arima_data = self.preprocessor.prepare_data_for_arima(df)
            
            # Buscar mejores parámetros usando grid search
            arima = ArimaPredictor()
            best_params = arima.grid_search_parameters(arima_data['returns'])
            
            # Actualizar modelo con los mejores parámetros
            arima.order = best_params[0]  # (p,d,q)
            arima.seasonal_order = best_params[1]  # (P,D,Q,s)
            
            # Entrenar con los mejores parámetros
            train_size = int(len(arima_data['returns']) * 0.8)
            train_data = {
                'returns': arima_data['returns'][:train_size],
                'close': arima_data['close'][:train_size]
            }
            test_data = {
                'returns': arima_data['returns'][train_size:],
                'close': arima_data['close'][train_size:]
            }
            
            arima.train(train_data)
            predictions = arima.predict(len(test_data['returns']))[0]
            
            # Evaluar resultados
            results = pd.DataFrame({
                'Real': test_data['returns'],
                'Predicción': predictions
            })
            
            metrics = self.evaluate_predictions(results)

            
        except Exception as e:
            self.logger.error(f"Error en prueba ARIMA: {str(e)}")
            print(traceback.format_exc())

    def evaluate_trading_performance(self, results: pd.DataFrame):
        """Evalúa el rendimiento desde perspectiva de trading"""
        try:
            # Señales de trading
            results['Signal'] = np.sign(results['Predicción'])
            results['Return'] = results['Real'] * results['Signal'].shift(1)
            
            # Métricas de trading
            total_trades = len(results[results['Signal'] != results['Signal'].shift(1)])
            winning_trades = len(results[results['Return'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Sharpe Ratio
            annual_factor = np.sqrt(252 * 24)  # Para datos horarios
            returns_mean = results['Return'].mean()
            returns_std = results['Return'].std()
            sharpe = returns_mean / returns_std * annual_factor if returns_std > 0 else 0
            
            # Maximum Drawdown
            cumulative_returns = (1 + results['Return']/100).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            print("\n=== Métricas de Trading ===")
            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Sharpe Ratio: {sharpe:.4f}")
            print(f"Maximum Drawdown: {max_drawdown:.2%}")
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Error en evaluación de trading: {str(e)}")
            raise

    def cross_validate_arima(self, data, n_splits=5):
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            metrics = []
            
            for train_idx, test_idx in tscv.split(data['returns']):
                train_data = {
                    'returns': data['returns'][train_idx],
                    'close': data['close'][train_idx]
                }
                test_data = {
                    'returns': data['returns'][test_idx],
                    'close': data['close'][test_idx]
                }
                
                arima = ArimaPredictor()
                arima.train(train_data)
                predictions = arima.predict(steps=len(test_idx))[0]
                
                fold_metrics = self.evaluate_predictions(pd.DataFrame({
                    'Real': test_data['returns'],
                    'Predicción': predictions
                }))
                metrics.append(fold_metrics)
                
            return pd.DataFrame(metrics).mean()
        except Exception as e:
            self.logger.error(f"Error en validación cruzada: {str(e)}")
            raise

def main():
    print("Iniciando pruebas de ARIMA...")
    tester = TestArima()
    tester.test_arima_prediction()

if __name__ == "__main__":
    main() 