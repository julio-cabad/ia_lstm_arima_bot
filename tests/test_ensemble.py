import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collector import DataCollector
from data.preprocessor import DataPreprocessor
from models.ensemble_model import EnsemblePredictor
from models.lstm_model import LSTMPredictor
from models.arima_model import ArimaPredictor
from utils.logger import setup_logger

class TestEnsemble:
    def __init__(self):
        self.logger = setup_logger("TestEnsemble")
        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()

    def test_ensemble_prediction(self):
        try:
            print("\nIniciando prueba de Ensemble...")
            
            # Configuración probada exitosamente
            symbol = "BTCUSDT"
            timeframe = "1h"
            df = self.collector.get_historical_data(symbol, timeframe)
            df = df.tail(2000)  # Mismo tamaño que tests exitosos
            
            # LSTM con parámetros probados
            X, y, scaler = self.preprocessor.prepare_data_for_lstm(df, look_back=20)
            train_size = int(len(X) * 0.8)  # 80-20 split probado
            
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            lstm = LSTMPredictor((X_train.shape[1], X_train.shape[2]))
            lstm.train(X_train, y_train)
            lstm_pred = lstm.predict(X_test).flatten()

            
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            lstm_pred = scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
            
            # Para LSTM
            results_lstm = pd.DataFrame({
                'Real': y_test,
                'Predicción': lstm_pred
            })
            print("\nMétricas LSTM:")
            lstm_metrics = self.evaluate_predictions(results_lstm)
            
            print(f"\nLongitud predicciones LSTM: {len(lstm_pred)}")
            print(f"Longitud y_test LSTM: {len(y_test)}")
            
            # Preparar y evaluar ARIMA
            print("\n=== Evaluación ARIMA ===")
            arima_data = self.preprocessor.prepare_data_for_arima(df)

            # Usar parámetros óptimos
            arima = ArimaPredictor()
            arima.order = (2, 1, 2)
            arima.seasonal_order = (1, 1, 2, 12)

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
            arima_pred = arima.predict(len(test_data['returns']))[0]

            # Para ARIMA (ya está correcto)
            results_arima = pd.DataFrame({
                'Real': test_data['returns'],
                'Predicción': arima_pred
            })
            print("\nMétricas ARIMA:")
            arima_metrics = self.evaluate_predictions(results_arima)

            print(f"\nLongitud predicciones ARIMA: {len(arima_pred)}")
            print(f"Longitud test_data ARIMA: {len(test_data['close'])}")
            
            # Asegurar misma longitud y mostrar
            min_len = min(len(lstm_pred), len(arima_pred), len(test_data['close']))
            print(f"\nUsando longitud mínima: {min_len}")
            
            lstm_pred = lstm_pred[:min_len]
            arima_pred = arima_pred[:min_len]
            test_close = test_data['close'][:min_len]
            
            print(f"Longitudes finales:")
            print(f"LSTM: {len(lstm_pred)}")
            print(f"ARIMA: {len(arima_pred)}")
            print(f"Test data: {len(test_close)}")
            
            # Escalar predicciones ARIMA
            arima_pred = scaler.transform(arima_pred.reshape(-1, 1)).flatten()
            arima_pred = scaler.inverse_transform(arima_pred.reshape(-1, 1)).flatten()

            # Usar mismo escalado para test_data
            test_close = scaler.transform(test_close.reshape(-1, 1)).flatten()
            test_close = scaler.inverse_transform(test_close.reshape(-1, 1)).flatten()
            
            # Ensemble con pesos optimizados
            total_error = lstm_metrics['mae'] + arima_metrics['mae']
            lstm_weight = arima_metrics['mae'] / total_error
            arima_weight = lstm_metrics['mae'] / total_error
            
            print(f"\nPesos del ensemble: LSTM={lstm_weight:.2f}, ARIMA={arima_weight:.2f}")
            ensemble_pred = lstm_pred * lstm_weight + arima_pred * arima_weight
            
            # Para Ensemble
            results_ensemble = pd.DataFrame({
                'Real': test_close,
                'Predicción': ensemble_pred
            })
            print("\nMétricas Ensemble:")
            self.evaluate_predictions(results_ensemble)
            
        except Exception as e:
            self.logger.error(f"Error en prueba ensemble: {str(e)}")
            raise 

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


def main():
    print("Iniciando pruebas de Ensemble...")
    tester = TestEnsemble()
    tester.test_ensemble_prediction()

if __name__ == "__main__":
    main() 
