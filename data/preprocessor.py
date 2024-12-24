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

    def prepare_data_for_lstm(self, df: pd.DataFrame, look_back: int = 30):
        try:
            # Instanciar Indicators
            indicators = Indicators(df)
            
            # Calcular indicadores usando la clase existente
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            
            # RSI
            df['rsi'] = indicators.rsi()
            
            # MACD
            macd_data = indicators.macd()
            df['macd'] = macd_data['MACD']
            df['signal'] = macd_data['Signal']
            df['macd_hist'] = macd_data['MACD_Hist']
            
            # Bollinger Bands
            bb_data = indicators.bollinger_bands(20)
            df['bb_upper'] = bb_data[0]
            df['bb_middle'] = bb_data[1]
            df['bb_lower'] = bb_data[2]
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # EMA Trend
            tm_color, tm_value = indicators.trend_magic()
            df['tm_value'] = tm_value
            
            # Codificación One-Hot para tm_color
            df['tm_color_blue'] = np.where(tm_color == 'blue', 1, 0)
            df['tm_color_red'] = np.where(tm_color == 'red', 1, 0)
            
            df['ema_slow'] = indicators.ema(21)
            df['ema_fast'] = indicators.ema(8)
            df['sma'] = indicators.sma(20)
            
            # Estrategia de cruce de medias móviles
            df['signal_cruce'] = 0
            df.loc[(df['ema_fast'].shift(1) < df['ema_slow'].shift(1)) & (df['ema_fast'] > df['ema_slow']), 'signal_cruce'] = 1
            df.loc[(df['ema_fast'].shift(1) > df['ema_slow'].shift(1)) & (df['ema_fast'] < df['ema_slow']), 'signal_cruce'] = -1
            
            # Nueva estrategia basada en precio, EMA y SMA
            df['signal_estrategia'] = 0
            df.loc[(df['close'] > df['ema_slow']) & (df['close'] > df['sma']), 'signal_estrategia'] = 1  # Long
            df.loc[(df['close'] < df['ema_slow']) & (df['close'] < df['sma']), 'signal_estrategia'] = -1 # Short
            
            # Nueva estrategia basada en precio y Magic Trend
            df['signal_magic_trend'] = 0
            df.loc[(df['close'] > df['tm_value']) & (df['tm_color_blue'] == 1), 'signal_magic_trend'] = 1  # Long
            df.loc[(df['close'] < df['tm_value']) & (df['tm_color_red'] == 1), 'signal_magic_trend'] = -1 # Short
            
            # Nueva estrategia basada en Bollinger Bands, Magic Trend y cruce de tm_value
            df['signal_bb_tm'] = 0
            
            # Calcular la distancia del tm_value a las bandas de Bollinger
            df['dist_to_lower'] = (df['tm_value'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['dist_to_upper'] = (df['bb_upper'] - df['tm_value']) / (df['bb_upper'] - df['bb_lower'])
            
            # Señal Long
            long_condition = (
                (df['tm_color_blue'] == 1) &
                (df['dist_to_lower'] < 0.3) &  # tm_value cerca de la banda inferior
                (df['dist_to_upper'] > 0.7) &  # tm_value lejos de la banda superior
                (df['tm_value'].shift(1) < df['bb_lower']) & (df['tm_value'] > df['bb_lower']) # Cruce de tm_value de abajo hacia arriba
            )
            
            # Señal Short
            short_condition = (
                (df['tm_color_red'] == 1) &
                (df['dist_to_upper'] < 0.3) &  # tm_value cerca de la banda superior
                (df['dist_to_lower'] > 0.7) &  # tm_value lejos de la banda inferior
                (df['tm_value'].shift(1) > df['bb_upper']) & (df['tm_value'] < df['bb_upper']) # Cruce de tm_value de arriba hacia abajo
            )
            
            df.loc[long_condition, 'signal_bb_tm'] = 1
            df.loc[short_condition, 'signal_bb_tm'] = -1
            
            # Nueva estrategia basada en el cruce del precio con tm_value, color y cercanía a la banda inferior
            df['signal_cruce_tm'] = 0
            
            # Señal Long
            long_cruce_condition = (
                (df['tm_color_blue'] == 1) &
                (df['dist_to_lower'] < 0.3) & # Precio cerca de la banda inferior
                (df['close'].shift(1) < df['tm_value']) & (df['close'] > df['tm_value']) & (df['close'] > df['tm_value']) # Cruce de abajo hacia arriba y precio mayor
            )
            
            # Señal Short
            short_cruce_condition = (
                (df['tm_color_red'] == 1) &
                (df['dist_to_upper'] < 0.3) & # Precio cerca de la banda superior
                (df['close'].shift(1) > df['tm_value']) & (df['close'] < df['tm_value']) & (df['close'] < df['tm_value']) # Cruce de arriba hacia abajo y precio menor
            )
            
            df.loc[long_cruce_condition, 'signal_cruce_tm'] = 1
            df.loc[short_cruce_condition, 'signal_cruce_tm'] = -1
            
            # Eliminar NaN
            df = df.dropna()
            
            # Features para el modelo
            features = np.column_stack((
                df['returns'].values * 100, # Retornos * 100
                df['signal_magic_trend'].values,
                df['signal_bb_tm'].values,
                df['signal_cruce_tm'].values,
                df['tm_color_red'].values,
                df['tm_color_blue'].values,
                df['tm_value'].values,
                df['signal_estrategia'].values,
                df['signal_cruce'].values,
                df['signal_cruce_tm'].values,
                df['signal_bb_tm'].values,
                df['macd_hist'].values,
                df['bb_width'].values,
                df['ema_slow'].values,
                df['ema_fast'].values,
                df['sma'].values
            ))
            
            # Normalizar features
            scaler = MinMaxScaler(feature_range=(0, 1))
            features_scaled = scaler.fit_transform(features)
            
            # Target: usar retornos futuros en lugar de dirección binaria
            target = df['close'].pct_change().shift(-1).values[:-1] * 100 # Retornos * 100
            
            # Crear secuencias
            X, y = [], []
            for i in range(look_back, len(features_scaled)-1):
                X.append(features_scaled[i-look_back:i])
                y.append(target[i])
            
            return np.array(X), np.array(y), scaler
            
        except Exception as e:
            self.logger.error(f"Error en prepare_data_for_lstm: {str(e)}")
            raise

   