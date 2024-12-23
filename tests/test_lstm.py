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

class TestLSTM:
    def __init__(self):
        self.logger = setup_logger("TestLSTM")
        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        
    def test_lstm_prediction(self):
        """Prueba el modelo LSTM"""
        try:
            print("\nIniciando prueba de LSTM...")
            
            # Obtener datos con timeframe más corto
            symbol = "BTCUSDT"
            timeframe = "30m"  # Cambiado de "1h" a "30m"
            print(f"\nObteniendo datos para {symbol} en timeframe {timeframe}")
            df = self.collector.get_historical_data(symbol, timeframe)
            
            # Limitar a datos más recientes pero aumentar la cantidad por el nuevo timeframe
            df = df.tail(4000)  # Duplicamos porque ahora tenemos el doble de registros por el timeframe
            
            # Preparar datos con ventana más corta
            print("\nPreparando datos...")
            X, y, scaler = self.preprocessor.prepare_data_for_lstm(df, look_back=20)  # Reducido de 30 a 20
            
            # Ajustar proporción de datos de entrenamiento
            train_size = int(len(X) * 0.7)  # Cambiado de 0.8 a 0.7
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            print(f"\nDatos de entrenamiento: {len(X_train)} registros")
            print(f"Datos de prueba: {len(X_test)} registros")
            
            # Obtener el número correcto de características
            n_features = X.shape[2]  # Obtener automáticamente el número de características
            input_shape = (60, n_features)
            lstm = LSTMPredictor(input_shape)
            
            # Entrenar modelo
            print("\nEntrenando modelo LSTM...")
            lstm.train(X_train, y_train, epochs=100, batch_size=32)
            
            # Realizar predicciones
            print("\nRealizando predicciones...")
            predictions = lstm.predict(X_test)
            
            # Convertir predicciones a precio real
            predictions_reshaped = np.zeros((len(predictions), n_features))  # Usar n_features en lugar de 6
            predictions_reshaped[:, 0] = predictions.flatten()  # Colocar predicciones en la primera columna
            predictions = scaler.inverse_transform(predictions_reshaped)[:, 0]  # Obtener solo la columna del precio
            
            # Preparar y_test para inverse_transform
            y_test_reshaped = np.zeros((len(y_test), n_features))  # Usar n_features en lugar de 6
            y_test_reshaped[:, 0] = y_test  # Colocar y_test en la primera columna
            y_test_actual = scaler.inverse_transform(y_test_reshaped)[:, 0]  # Obtener solo la columna del precio
            
            # Evaluar modelo
            print("\nEvaluando modelo...")
            mse = mean_squared_error(y_test_actual, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_actual, predictions)
            
            print("\n=== Métricas de Evaluación ===")
            print(f"MSE: {mse:.6f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {mae:.6f}")
            
            # Visualizar resultados
            self.plot_predictions(df['close'].values[-len(predictions):], predictions)
            
            print("\nPrueba completada exitosamente")
            
        except Exception as e:
            self.logger.error(f"Error en prueba LSTM: {str(e)}")
            print(traceback.format_exc())
            
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
            
    def evaluate_predictions(self, y_true: np.ndarray, predictions: np.ndarray):
        """Evaluación detallada de predicciones"""
        try:
            # Métricas básicas
            mse = mean_squared_error(y_true, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, predictions)
            
            # Métricas direccionales
            direction_true = np.sign(np.diff(y_true))
            direction_pred = np.sign(np.diff(predictions.flatten()))
            direction_accuracy = np.mean(direction_true == direction_pred)
            
            print("\n=== Métricas de Evaluación ===")
            print(f"MSE: {mse:.6f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"Precisión Direccional: {direction_accuracy:.2%}")
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'direction_accuracy': direction_accuracy
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