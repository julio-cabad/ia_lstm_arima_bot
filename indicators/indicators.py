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

    def rsi(self, period=14):
        try:
            return talib.RSI(self.close, timeperiod=period).iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculando RSI: {str(e)}")
            raise

    def macd(self):
        try:
            macd, signal, hist = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9)
            return {
                'MACD': macd.iloc[-1],
                'Signal': signal.iloc[-1],
                'MACD_Hist': hist.iloc[-1]
            }
        except Exception as e:
            self.logger.error(f"Error calculando MACD: {str(e)}")
            raise

    def bollinger_bands(self, timeperiod: int):
        upper, middle, lower = talib.BBANDS(self.close, timeperiod=timeperiod, nbdevup=2, nbdevdn=2, matype=0)
        return upper.iloc[-1], middle.iloc[-1], lower.iloc[-1]

    def ema(self, timeperiod: int):
        return round(talib.EMA(self.close, timeperiod=timeperiod).iloc[-1], 5)
        
    def sma(self, timeperiod: int):
        return round(talib.SMA(self.close, timeperiod=timeperiod).iloc[-1], 5)

    def trend_magic(self):
        color = 'green'
        cci = None
        period = 20
        coeff = 1
        ap = 5

        if cci is None:
            cci = talib.CCI(self.high, self.low, self.close, timeperiod=period)

        tr = talib.ATR(self.high, self.low, self.close, timeperiod=ap)

        up = self.low - tr * coeff
        down = self.high + tr * coeff

        magic_trend = pd.Series([0.0] * len(self.data))

        for i in range(len(self.data)):
            # Define el color de la línea MagicTrend.
            color = 'blue' if cci[i] > 0 else 'red'

            if cci[i] >= 0:
                if not math.isnan(up[i]):
                    magic_trend[i] = up[i] if i == 0 else max(up[i], magic_trend[i - 1])
                else:
                    magic_trend[i] = magic_trend[i - 1] if i > 0 else np.nan
            else:
                if not math.isnan(down[i]):
                    magic_trend[i] = down[i] if i == 0 else min(down[i], magic_trend[i - 1])
                else:
                    magic_trend[i] = magic_trend[i - 1] if i > 0 else np.nan

        mt = magic_trend.iloc[-1], 2

        if mt is not None:
            return color, round(magic_trend.iloc[-1], 3)
