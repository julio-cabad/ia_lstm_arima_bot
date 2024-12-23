import pandas as pd
import numpy as np
from typing import Dict, Tuple
from utils.logger import setup_logger
from indicators.indicators import Indicators
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

class DataPreprocessor:
    def __init__(self):
        self.logger = setup_logger("DataPreprocessor")
        
    def prepare_data_for_arima(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        try:
            data = df.copy()
            
            # Calcular retornos logarítmicos
            data['returns'] = np.log(data['close']).diff() * 100
            
            # Eliminar NaN antes de calcular z-scores
            data = data.dropna()
            
            # Eliminar outliers usando máscara
            z_scores = np.abs(stats.zscore(data['returns']))
            data = data[z_scores < 3]
            
            # Características adicionales
            data['volatility'] = data['close'].pct_change().rolling(20).std()
            data['momentum'] = data['close'].pct_change(5).rolling(5).mean()
            data['trend'] = data['close'].rolling(50).mean().pct_change()
            
            # Eliminar valores nulos finales
            data = data.dropna()
            
            return {
                'returns': data['returns'].values,
                'dates': data.index,
                'close': data['close'].values,
                'volatility': data['volatility'].values,
                'momentum': data['momentum'].values,
                'trend': data['trend'].values
            }
        except Exception as e:
            self.logger.error(f"Error en preprocesamiento: {str(e)}")
            raise

    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """Valida la calidad de los datos"""
        try:
            print("\n=== Validación de Calidad de Datos ===")
            
            # 1. Verificar suficientes datos
            min_required = 100
            print(f"\nDatos disponibles: {len(data)} (mínimo requerido: {min_required})")
            if len(data) < min_required:
                self.logger.warning(f"Insuficientes datos: {len(data)} < {min_required}")
                return False
            
            # 2. Verificar estacionariedad
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(data['returns'])
            print(f"\nTest de Estacionariedad (ADF):")
            print(f"p-value: {adf_result[1]:.4f}")
            
            if adf_result[1] > 0.05:
                print("Los datos no son estacionarios, aplicando diferenciación...")
                data['returns'] = np.diff(data['returns'], prepend=data['returns'][0])
                adf_result = adfuller(data['returns'])
                print(f"p-value después de diferenciación: {adf_result[1]:.4f}")
            
            # 3. Verificar autocorrelación
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(data['returns'], lags=[10])
            print(f"\nTest de Autocorrelación (Ljung-Box):")
            print(f"p-value: {lb_result['lb_pvalue'].iloc[0]:.4f}")
            
            if lb_result['lb_pvalue'].iloc[0] > 0.05:
                print("Advertencia: Baja autocorrelación en los datos")
            
            # 4. Verificar volatilidad
            from arch import arch_model
            am = arch_model(data['returns'])
            res = am.fit(disp='off')
            arch_test = res.arch_lm_test()
            print(f"\nTest de Volatilidad:")
            try:
                print(f"Test Statistic: {float(arch_test.stat):.4f}")
                print(f"p-value: {float(arch_test.pvalue):.4f}")
            except:
                print(f"Test Statistic: {arch_test.stat}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validando datos: {str(e)}")
            print(f"Error durante la validación: {str(e)}")
            return True

    def prepare_data_for_lstm(self, df: pd.DataFrame, look_back: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        try:
            # Usar solo el precio de cierre
            feature_data = df[['close']].copy()
            
            # Normalizar datos
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(feature_data)
            
            # Crear secuencias
            X, y = [], []
            for i in range(look_back, len(scaled_data)):
                X.append(scaled_data[i-look_back:i])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            
            # Asegurar la forma correcta para LSTM
            X = np.reshape(X, (X.shape[0], look_back, 1))
            
            # Logging para debugging
            self.logger.info(f"Shape de X: {X.shape}")
            self.logger.info(f"Shape de y: {y.shape}")
            
            return X, y, scaler
            
        except Exception as e:
            self.logger.error(f"Error en prepare_data_for_lstm: {str(e)}")
            raise

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, lower