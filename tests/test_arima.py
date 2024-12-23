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
            
            # Métricas direccionales mejoradas
            correct_direction = np.mean(
                (results['Real'] * results['Predicción']) > 0
            )
            
            # Métricas de trading
            results['Signal'] = np.sign(results['Predicción'])
            results['Return'] = results['Real'] * results['Signal'].shift(1)
            
            win_rate = len(results[results['Return'] > 0]) / len(results[results['Return'] != 0])
            sharpe = np.sqrt(252) * results['Return'].mean() / results['Return'].std()
            
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
            
            # Obtener y preparar datos iniciales
            symbol = "BTCUSDT"
            timeframe = "1h"
            df = self.collector.get_historical_data(symbol, timeframe)
            df = df.tail(2000)
            arima_data = self.preprocessor.prepare_data_for_arima(df)
            
            # Ya no necesitamos buscar parámetros
            # best_params = arima.grid_search_parameters(arima_data['returns'])
            
            # Validación cruzada directa con parámetros óptimos
            print("\nRealizando validación cruzada...")
            cv_metrics = self.cross_validate_arima(arima_data)
            print("\nMétricas promedio en validación cruzada:")
            for metric, value in cv_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Continuar con el entrenamiento y evaluación final...
            # (resto del código existente)
            
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