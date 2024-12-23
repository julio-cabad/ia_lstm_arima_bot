from typing import Dict
import pandas as pd
from utils.logger import setup_logger
from bnc.binance import RobotBinance

class DataCollector:
    def __init__(self):
        self.logger = setup_logger("DataCollector")
        
    def get_historical_data(self, symbol: str, timeframe: str, limit: int = 5000) -> pd.DataFrame:
        """
        Obtiene datos históricos de Binance para un símbolo específico.
        
        Args:
            symbol: Par de trading (ej: 'BTCUSDT')
            timeframe: Intervalo de tiempo (ej: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d')
            limit: Número máximo de velas históricas a obtener
            
        Returns:
            DataFrame con los datos históricos
        """
        try:
            # Validar timeframe
            valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
            if timeframe not in valid_timeframes:
                raise ValueError(f"Timeframe inválido. Debe ser uno de: {valid_timeframes}")
            
            client = RobotBinance(pair=symbol, temporality=timeframe)
            df = client.candlestick()
            
            # Asegurar que el índice temporal sea correcto
            df.index = pd.to_datetime(df.index, unit='ms')
            
            # Verificar cantidad de datos
            if len(df) < limit:  # 90% del objetivo
                self.logger.warning(f"Obtenidos menos datos de los esperados: {len(df)} < {limit}")
            
            self.logger.info(f"Successfully collected {len(df)} records for {symbol}")
            
            # Información adicional
            print(f"\nPeríodo de datos:")
            print(f"Desde: {df.index.min()}")
            print(f"Hasta: {df.index.max()}")
            print(f"Total registros: {len(df)}")
            
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