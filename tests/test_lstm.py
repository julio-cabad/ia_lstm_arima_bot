import os
# Configurar el uso de CPU al inicio del archivo
os.environ['TENSORFLOW_METAL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collector import DataCollector
from data.preprocessor import DataPreprocessor
from models.lstm_model import LSTMPredictor
from utils.logger import setup_logger
from indicators.indicators import Indicators

class TestLSTM:
    def __init__(self):
        self.logger = setup_logger("TestLSTM")
        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        
    def test_lstm_prediction(self):
        try:
            print("\nIniciando prueba de LSTM...")
            
            # 1. Obtener datos
            symbol = "BTCUSDT"
            timeframe = "4h"
            df = self.collector.get_historical_data(symbol, timeframe)
            df = df.tail(2000)
            
            # Guardar los precios originales
            df_prices = df['close'].copy()
            
            # 2. Preparar datos
            X, y, scaler = self.preprocessor.prepare_data_for_lstm(df, look_back=30)
            print(f"\nShape de datos:")
            print(f"X: {X.shape}")
            print(f"y: {y.shape}")
            
            # 3. Split train/test
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # 4. Entrenar modelo
            lstm = LSTMPredictor((X_train.shape[1], X_train.shape[2]))
            lstm.train(X_train, y_train)
            
            # 5. Predecir
            predictions = lstm.predict(X_test)
            
            # Crear DataFrame de resultados con precios reales
            results = pd.DataFrame({
                'Real': df_prices[-(len(y_test)):].values,  # Precios reales
                'Predicción': predictions.flatten()  # Retornos predichos
            })
            
            # Limpiar datos antes de calcular retornos
            results = results.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Calcular retornos reales
            results['Returns'] = results['Real'].pct_change()
            
            # 7. Visualizar
            if len(results) > 1:
                plt.figure(figsize=(15,7))
                plt.plot(results.index, results['Returns'], label='Retornos Reales', alpha=0.5)
                plt.plot(results.index, results['Predicción'], label='Retornos Predichos', alpha=0.5)
                plt.title('LSTM: Predicción de Retornos')
                plt.legend()
                plt.savefig('tests/lstm_predictions.png')
                plt.close()
            else:
                self.logger.warning("No hay suficientes datos para visualizar.")
            
            if len(results) > 1:
                return self.evaluate_predictions(results)
            else:
                self.logger.warning("No hay suficientes datos para evaluar.")
                return {}
            
        except Exception as e:
            self.logger.error(f"Error en prueba LSTM: {str(e)}")
            raise
            
    def plot_predictions(self, actual: np.ndarray, predictions: np.ndarray):
        """Visualiza las predicciones vs valores reales"""
        try:
            plt.figure(figsize=(15, 7))
            plt.plot(actual, color='blue', label='Real')
            plt.plot(predictions, color='red', label='Predicción')
            plt.title('Predicciones LSTM vs Valores Reales')
            plt.xlabel('Tiempo')
            plt.ylabel('Precio')
            plt.legend()
            plt.grid(True)
            plt.savefig('tests/lstm_predictions.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error en visualización: {str(e)}")
            raise
            
    def evaluate_predictions(self, results: pd.DataFrame) -> dict:
        """Evaluación detallada de predicciones de retornos"""
        try:
            # Limpiar datos
            results = results.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(results) < 2:
                self.logger.warning("No hay suficientes datos para evaluar.")
                return {}
            
            # Métricas de error sobre retornos
            mse = mean_squared_error(results['Returns'], results['Predicción'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(results['Returns'], results['Predicción'])
            
            # Métricas de dirección
            results['Real_Direction'] = np.sign(results['Returns'])
            results['Pred_Direction'] = np.sign(results['Predicción'])
            accuracy = np.mean(results['Real_Direction'] == results['Pred_Direction'])
            
            # Señales de trading
            results['Signal'] = 0
            results.loc[results['Predicción'] > 0.001, 'Signal'] = 1    # Long si predice subida >0.1%
            results.loc[results['Predicción'] < -0.001, 'Signal'] = -1  # Short si predice bajada >0.1%
            
            # Retornos de la estrategia
            results['Strategy_Returns'] = results['Returns'] * results['Signal'].shift(1)
            
            # Métricas de trading
            win_rate = np.mean(results['Strategy_Returns'] > 0)
            
            # Sharpe Ratio anualizado
            annual_factor = np.sqrt(252)
            returns_mean = results['Strategy_Returns'].mean()
            returns_std = results['Strategy_Returns'].std()
            sharpe = annual_factor * (returns_mean / returns_std) if returns_std != 0 else 0
            
            print("\n=== Métricas de Evaluación ===")
            print(f"MSE (Retornos): {mse:.6f}")
            print(f"RMSE (Retornos): {rmse:.6f}")
            print(f"MAE (Retornos): {mae:.6f}")
            print(f"Precisión Direccional: {accuracy:.2%}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Retorno Medio Diario: {returns_mean:.4%}")
            print(f"Volatilidad Diaria: {returns_std:.4%}")
            print(f"Sharpe Ratio: {sharpe:.4f}")
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'accuracy': accuracy,
                'win_rate': win_rate,
                'returns_mean': returns_mean,
                'returns_std': returns_std,
                'sharpe_ratio': sharpe
            }
            
        except Exception as e:
            self.logger.error(f"Error en evaluación: {str(e)}")
            raise

def main():
    print("Iniciando pruebas de LSTM...")
    tester = TestLSTM()
    tester.test_lstm_prediction()

if __name__ == "__main__":
    main() 