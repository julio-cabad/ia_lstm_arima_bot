import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback

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
        
    def evaluate_predictions(self, results: pd.DataFrame):
        """Evaluación detallada de predicciones"""
        try:
            # Métricas básicas
            mse = mean_squared_error(results['Real'], results['Predicción'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(results['Real'], results['Predicción'])
            
            # Métricas direccionales
            correct_direction = np.mean(np.sign(results['Real']) == np.sign(results['Predicción']))
            
            # Métricas de error en eventos extremos
            extreme_mask = np.abs(results['Real']) > np.std(results['Real'])
            extreme_rmse = np.sqrt(mean_squared_error(
                results.loc[extreme_mask, 'Real'],
                results.loc[extreme_mask, 'Predicción']
            ))
            
            print("\n=== Evaluación Detallada ===")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"Dirección Correcta: {correct_direction:.2%}")
            print(f"RMSE en eventos extremos: {extreme_rmse:.4f}")
            
            return {
                'rmse': rmse,
                'mae': mae,
                'correct_direction': correct_direction,
                'extreme_rmse': extreme_rmse
            }
            
        except Exception as e:
            self.logger.error(f"Error en evaluación: {str(e)}")
            raise
            
    def plot_detailed_results(self, results: pd.DataFrame):
        """Visualización detallada de resultados"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
            
            # Plot 1: Predicciones vs Reales
            ax1.plot(results.index, results['Real'], label='Real', color='blue', alpha=0.8)
            ax1.plot(results.index, results['Predicción'], label='Predicción', color='red', alpha=0.8)
            ax1.fill_between(results.index, 
                            results['Límite_Inferior'],
                            results['Límite_Superior'],
                            color='gray', alpha=0.2)
            ax1.set_title('Predicciones vs Valores Reales')
            ax1.legend()
            
            # Plot 2: Error de predicción
            error = results['Real'] - results['Predicción']
            ax2.plot(results.index, error, color='green', alpha=0.8)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_title('Error de Predicción')
            
            # Plot 3: Distribución del error
            sns.histplot(error, ax=ax3, bins=50)
            ax3.axvline(x=0, color='r', linestyle='--')
            ax3.set_title('Distribución del Error')
            
            plt.tight_layout()
            plt.savefig('tests/arima_detailed_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error en visualización: {str(e)}")
            raise

    def test_arima_prediction(self):
        """Prueba el modelo ARIMA"""
        try:
            print("\nIniciando prueba de ARIMA...")
            
            # Obtener datos
            symbol = "BTCUSDT"
            timeframe = "1h"
            print(f"\nObteniendo datos para {symbol} en timeframe {timeframe}")
            df = self.collector.get_historical_data(symbol, timeframe)
            
            print(f"\nDatos totales obtenidos: {len(df)} registros")
            
            if df.empty:
                self.logger.error("No se pudieron obtener datos para las pruebas")
                return
            
            # Preparar datos
            print("\nPreparando datos...")
            arima_data = self.preprocessor.prepare_data_for_arima(df)
            print(f"Datos después de preprocesamiento: {len(arima_data['returns'])} registros")
            
            # Dividir datos en entrenamiento y prueba
            train_size = int(len(arima_data['returns']) * 0.8)
            train_data = {
                'returns': arima_data['returns'][:train_size],
                'close': arima_data['close'][:train_size]
            }
            test_data = {
                'returns': arima_data['returns'][train_size:],
                'close': arima_data['close'][train_size:]
            }
            
            # Verificar datos antes de entrenar
            min_samples = 50  # Reducir el mínimo de muestras requeridas
            if len(train_data['returns']) < min_samples:
                raise ValueError(f"Insuficientes datos para entrenamiento. Se necesitan al menos {min_samples} registros")

            print(f"\nDatos de entrenamiento: {len(train_data['returns'])} registros")
            print(f"Datos de prueba: {len(test_data['returns'])} registros")
            
            # Entrenar modelo
            print("\nEntrenando modelo ARIMA...")
            arima = ArimaPredictor()  # Usará los órdenes predefinidos
            arima.train(train_data)
            
            # Realizar predicciones
            print("\nRealizando predicciones...")
            predictions, conf_int = arima.predict(steps=len(test_data['returns']))
            
            # Evaluar modelo
            print("\nEvaluando modelo...")
            metrics = arima.evaluate(test_data['returns'])
            
            # Guardar y visualizar resultados
            results = pd.DataFrame({
                'Fecha': arima_data['dates'][train_size:],
                'Real': test_data['returns'],
                'Predicción': predictions,
                'Límite_Inferior': conf_int[:,0],
                'Límite_Superior': conf_int[:,1]
            })
            results.set_index('Fecha', inplace=True)
            results.to_csv('tests/arima_results.csv')
            
            # Evaluación detallada
            self.evaluate_predictions(results)
            
            # Generar gráfica
            self.plot_detailed_results(results)
            
            print("\nPrueba completada exitosamente")
            
        except Exception as e:
            self.logger.error(f"Error en prueba ARIMA: {str(e)}")
            import traceback
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

    def test_lstm_prediction(self):
        try:
            # Obtener y preparar datos
            symbol = "BTCUSDT"
            timeframe = "4h"
            df = self.collector.get_historical_data(symbol, timeframe)
            X, y, scaler = self.preprocessor.prepare_data_for_lstm(df)

            # Inicializar y entrenar LSTM
            lstm = LSTMPredictor((X.shape[1], X.shape[2]))
            lstm.train(X, y)

            # Realizar predicciones
            predictions = lstm.predict(X)
            predictions = scaler.inverse_transform(predictions)

            # Visualizar resultados
            plt.figure(figsize=(14,5))
            plt.plot(df['close'].values, color='blue', label='Real')
            plt.plot(range(len(y), len(y) + len(predictions)), predictions, color='red', label='Predicción')
            plt.title('Predicción LSTM')
            plt.xlabel('Tiempo')
            plt.ylabel('Precio')
            plt.legend()
            plt.show()
        
        except Exception as e:
            self.logger.error(f"Error en prueba LSTM: {str(e)}")
            print(traceback.format_exc())

def main():
    print("Iniciando pruebas de ARIMA...")
    tester = TestArima()
    tester.test_arima_prediction()

if __name__ == "__main__":
    main() 