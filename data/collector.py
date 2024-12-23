from typing import Dict
import pandas as pd
from utils.logger import setup_logger
from bnc.binance import RobotBinance

class DataCollector:
    def __init__(self):
        self.logger = setup_logger("DataCollector")
        
    def get_historical_data(self, symbol: str, timeframe: str, lookback_days: int = 30) -> pd.DataFrame:
        """
        Obtiene datos históricos de Binance para un símbolo específico.
        
        Args:
            symbol: Par de trading (ej: 'BTCUSDT')
            timeframe: Intervalo de tiempo (ej: '1h', '4h', '1d')
            lookback_days: Número de días hacia atrás para obtener datos
            
        Returns:
            DataFrame con los datos históricos
        """
        try:
            client = RobotBinance(pair=symbol, temporality=timeframe)
            df = client.candlestick()
            
            # Asegurar que el índice temporal sea correcto
            df.index = pd.to_datetime(df.index, unit='ms')
            
            self.logger.info(f"Successfully collected {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    def get_recent_trades(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """
        Obtiene trades recientes para análisis de volumen y momentum
        """
        try:
            client = RobotBinance(pair=symbol, temporality="1m")  # temporality no importa para trades
            trades = client.get_recent_trades(limit=limit)
            
            df = pd.DataFrame(trades)
            if not df.empty and 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting recent trades for {symbol}: {str(e)}")
            return pd.DataFrame() 