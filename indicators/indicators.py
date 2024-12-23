import math
import talib
import numpy as np
import pandas as pd
import pandas_ta as ta
from utils.logger import setup_logger

class Indicators:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']
        self.open = data['open']
        self.volume = data['volume']
        self.logger = setup_logger("Indicators")
        self.data.ta.cores = 0  # Desactivar multiprocesamiento para más control
        
    def add_all_indicators(self):
        """Añade todos los indicadores al DataFrame"""
        try:
            print("\nAñadiendo VWAP...")
            self.data.ta.vwap(append=True)
            
            print("\nCreando estrategia personalizada...")
            custom_strategy = ta.Strategy(
                name="custom_strategy",
                description="Estrategia con indicadores personalizados",
                ta=[
                    # Tendencia
                    {"kind": "ema", "length": 9},
                    {"kind": "ema", "length": 21},
                    {"kind": "supertrend", "length": 10, "multiplier": 3},
                    {"kind": "adx", "length": 14},
                    
                    # Momentum
                    {"kind": "rsi", "length": 14},
                    {"kind": "macd", "fast": 8, "slow": 21, "signal": 9},
                    {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3},
                    
                    # Volatilidad
                    {"kind": "bbands", "length": 20},
                    {"kind": "atr", "length": 14},
                    {"kind": "kc", "length": 20, "scalar": 2},
                    
                    # Volumen
                    {"kind": "mfi", "length": 14},
                    {"kind": "obv"},
                    {"kind": "cmf", "length": 20}
                ]
            )
            
            print("\nAplicando estrategia...")
            self.data.ta.strategy(custom_strategy)
            
            print("\nCalculando RSX...")
            rsx_series = self.calculate_rsx_series(length=14)
            self.data['RSX_14'] = rsx_series
            
            print("\nCalculando TrendMagic...")
            magic_trend, magic_colors = self.calculate_trend_magic_series()
            self.data['TREND_MAGIC'] = magic_trend
            self.data['TREND_MAGIC_COLOR'] = magic_colors
            
            print("\nTodos los indicadores añadidos exitosamente")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error añadiendo indicadores: {str(e)}")
            print(f"\nError añadiendo indicadores: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return self.data

    def calculate_rsx_series(self, length=14):
        """Calcula RSX para toda la serie temporal de una vez"""
        try:
            df = self.data.copy()
            
            # --- Preparación de la fuente ---
            if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
                df['src'] = (df['high'] + df['low'] + df['close']) / 3
            elif 'close' in df.columns:
                df['src'] = df['close']
            else:
                raise ValueError("El DataFrame debe contener 'close' o 'high', 'low' y 'close' para calcular la fuente (source).")

            f8 = 100 * df['src']
            f10 = f8.shift(1).fillna(0)
            v8 = f8 - f10
            
            f18 = 3 / (length + 2)
            f20 = 1 - f18

            f28 = pd.Series(np.zeros(len(df)), index=df.index)
            f30 = pd.Series(np.zeros(len(df)), index=df.index)
            f38 = pd.Series(np.zeros(len(df)), index=df.index)
            f40 = pd.Series(np.zeros(len(df)), index=df.index)
            f48 = pd.Series(np.zeros(len(df)), index=df.index)
            f50 = pd.Series(np.zeros(len(df)), index=df.index)
            f58 = pd.Series(np.zeros(len(df)), index=df.index)
            f60 = pd.Series(np.zeros(len(df)), index=df.index)
            f68 = pd.Series(np.zeros(len(df)), index=df.index)
            f70 = pd.Series(np.zeros(len(df)), index=df.index)
            f78 = pd.Series(np.zeros(len(df)), index=df.index)
            f80 = pd.Series(np.zeros(len(df)), index=df.index)
                
            for i in range(1, len(df)):
                f28.iloc[i] = f20 * f28.iloc[i-1] + f18 * v8.iloc[i]
                f30.iloc[i] = f18 * f28.iloc[i] + f20 * f30.iloc[i-1]
            vC = f28 * 1.5 - f30 * 0.5

            for i in range(1, len(df)):
                f38.iloc[i] = f20 * f38.iloc[i-1] + f18 * vC.iloc[i]
                f40.iloc[i] = f18 * f38.iloc[i] + f20 * f40.iloc[i-1]
            v10 = f38 * 1.5 - f40 * 0.5
            
            for i in range(1, len(df)):
                f48.iloc[i] = f20 * f48.iloc[i-1] + f18 * v10.iloc[i]
                f50.iloc[i] = f18 * f48.iloc[i] + f20 * f50.iloc[i-1]
            v14 = f48 * 1.5 - f50 * 0.5
            
            for i in range(1, len(df)):
                f58.iloc[i] = f20 * f58.iloc[i-1] + f18 * abs(v8.iloc[i])
                f60.iloc[i] = f18 * f58.iloc[i] + f20 * f60.iloc[i-1]
            v18 = f58 * 1.5 - f60 * 0.5
            
            for i in range(1, len(df)):
                f68.iloc[i] = f20 * f68.iloc[i-1] + f18 * v18.iloc[i]
                f70.iloc[i] = f18 * f68.iloc[i] + f20 * f70.iloc[i-1]
            v1C = f68 * 1.5 - f70 * 0.5
            
            for i in range(1, len(df)):
                f78.iloc[i] = f20 * f78.iloc[i-1] + f18 * v1C.iloc[i]
                f80.iloc[i] = f18 * f78.iloc[i] + f20 * f80.iloc[i-1]
            v20 = f78 * 1.5 - f80 * 0.5

            f88 = pd.Series(np.zeros(len(df)), index=df.index)
            f90 = pd.Series(np.zeros(len(df)), index=df.index)
            f90_ = pd.Series(np.zeros(len(df)), index=df.index)

            f88.iloc[0] = length - 1 if length - 1 >= 5 else 5

            for i in range(1, len(df)):
                if f90_.iloc[i-1] == 0:
                    f90_.iloc[i] = 1
                elif f88.iloc[i-1] <= f90_.iloc[i-1]:
                    f90_.iloc[i] = f88.iloc[i-1] + 1
                else:
                    f90_.iloc[i] = f90_.iloc[i-1] + 1
                    
                if f90_.iloc[i-1] == 0 and (length - 1 >= 5):
                    f88.iloc[i] = length - 1
                else:
                    f88.iloc[i] = 5
                    
            f0 = (f88 >= f90_) & (f8 != f10)
            f90 = np.where((f88 == f90_) & (~f0), 0, f90_)

            v4_ = np.where((f88 < f90) & (v20 > 0), (v14 / v20 + 1) * 50, 50)

            rsx = np.clip(v4_, 0, 100)
            return pd.Series(rsx, index=df.index)
                
        except Exception as e:
            self.logger.error(f"Error calculando RSX series: {str(e)}")
            return pd.Series(np.nan, index=self.data.index)

    def calculate_trend_magic_series(self):
        """Calcula TrendMagic para toda la serie temporal de una vez"""
        try:
            cci = talib.CCI(self.high, self.low, self.close, timeperiod=20)
            tr = talib.ATR(self.high, self.low, self.close, timeperiod=5)
            
            up = self.low - tr
            down = self.high + tr
            
            magic_trend = pd.Series(0.0, index=self.data.index)
            colors = pd.Series('green', index=self.data.index)
            
            for i in range(len(self.data)):
                colors.iloc[i] = 'blue' if cci.iloc[i] > 0 else 'red'
                
                if cci.iloc[i] >= 0:
                    if not math.isnan(up.iloc[i]):
                        magic_trend.iloc[i] = up.iloc[i] if i == 0 else max(up.iloc[i], magic_trend.iloc[i - 1])
                    else:
                        magic_trend.iloc[i] = magic_trend.iloc[i - 1] if i > 0 else np.nan
                else:
                    if not math.isnan(down.iloc[i]):
                        magic_trend.iloc[i] = down.iloc[i] if i == 0 else min(down.iloc[i], magic_trend.iloc[i - 1])
                    else:
                        magic_trend.iloc[i] = magic_trend.iloc[i - 1] if i > 0 else np.nan
            
            return magic_trend.round(3), colors
                
        except Exception as e:
            self.logger.error(f"Error calculando TrendMagic series: {str(e)}")
            return pd.Series(np.nan, index=self.data.index), pd.Series('none', index=self.data.index)
