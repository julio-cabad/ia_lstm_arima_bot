import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collector import DataCollector
from indicators.indicators import Indicators
from utils.logger import setup_logger

class TestIndicators:
    def __init__(self):
        self.logger = setup_logger("TestIndicators")
        self.collector = DataCollector()
        
    def test_all_indicators(self):
        """Prueba todos los indicadores"""
        try:
            print("\nIniciando prueba de indicadores...")
            
            # Obtener datos de prueba
            symbol = "BTCUSDT"
            timeframe = "1h"
            print(f"\nObteniendo datos para {symbol} en timeframe {timeframe}")
            df = self.collector.get_historical_data(symbol, timeframe)
            
            if df.empty:
                self.logger.error("No se pudieron obtener datos para las pruebas")
                return
                
            print(f"\nDatos obtenidos: {len(df)} registros")
            
            # Verificar que el índice sea temporal para VWAP
            if not isinstance(df.index, pd.DatetimeIndex):
                print("\nConvirtiendo índice a DatetimeIndex")
                df.index = pd.to_datetime(df.index)
                print("Índice convertido exitosamente")
            
            # Inicializar indicadores
            print("\nInicializando indicadores...")
            indicators = Indicators(df)
            
            # Aplicar todos los indicadores
            print("\nAplicando indicadores...")
            df = indicators.add_all_indicators()
            print("Indicadores aplicados exitosamente")
            
            # Mostrar información general
            print(f"\nTotal de columnas: {len(df.columns)}")
            print(f"Período de tiempo: {df.index[0]} a {df.index[-1]}")
            
            # Verificar los resultados
            print("\nVerificando resultados...")
            self.verify_trend_indicators(df)
            self.verify_momentum_indicators(df)
            self.verify_volatility_indicators(df)
            self.verify_volume_indicators(df)
            self.verify_custom_indicators(df)
            
            # Guardar resultados en CSV para inspección
            print("\nGuardando resultados en CSV...")
            df.to_csv('tests/indicator_results.csv')
            print("Pruebas completadas exitosamente")
            
        except Exception as e:
            self.logger.error(f"Error en pruebas de indicadores: {str(e)}")
            print(f"\nError durante las pruebas: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def verify_trend_indicators(self, df):
        """Verifica indicadores de tendencia"""
        try:
            print("\n=== Indicadores de Tendencia ===")
            trend_columns = [col for col in df.columns if any(x in col.lower() 
                for x in ['ema', 'supertrend', 'adx'])]
            print("Columnas encontradas:", trend_columns)
            if trend_columns:
                print("\nÚltimos valores:")
                print(df[trend_columns].tail(1))
        except Exception as e:
            self.logger.error(f"Error verificando indicadores de tendencia: {str(e)}")
    
    def verify_momentum_indicators(self, df):
        """Verifica indicadores de momentum"""
        try:
            print("\n=== Indicadores de Momentum ===")
            momentum_columns = [col for col in df.columns if any(x in col.lower() 
                for x in ['rsi', 'macd', 'stoch'])]
            print("Columnas encontradas:", momentum_columns)
            if momentum_columns:
                print("\nÚltimos valores:")
                print(df[momentum_columns].tail(1))
        except Exception as e:
            self.logger.error(f"Error verificando indicadores de momentum: {str(e)}")
    
    def verify_volatility_indicators(self, df):
        """Verifica indicadores de volatilidad"""
        try:
            print("\n=== Indicadores de Volatilidad ===")
            volatility_columns = [col for col in df.columns if any(x in col.lower() 
                for x in ['bb_', 'atr', 'kc_'])]
            print("Columnas encontradas:", volatility_columns)
            if volatility_columns:
                print("\nÚltimos valores:")
                print(df[volatility_columns].tail(1))
        except Exception as e:
            self.logger.error(f"Error verificando indicadores de volatilidad: {str(e)}")
    
    def verify_volume_indicators(self, df):
        """Verifica indicadores de volumen"""
        try:
            print("\n=== Indicadores de Volumen ===")
            volume_columns = [col for col in df.columns if any(x in col.upper() 
                for x in ['VWAP', 'MFI', 'OBV', 'CMF'])]
            print("Columnas encontradas:", volume_columns)
            if volume_columns:
                print("\nÚltimos valores:")
                print(df[volume_columns].tail(1))
        except Exception as e:
            self.logger.error(f"Error verificando indicadores de volumen: {str(e)}")
    
    def verify_custom_indicators(self, df):
        """Verifica indicadores personalizados"""
        try:
            print("\n=== Indicadores Personalizados ===")
            custom_columns = [col for col in df.columns if any(x in col.upper() 
                for x in ['TREND_MAGIC', 'RSX'])]
            print("Columnas encontradas:", custom_columns)
            if custom_columns:
                print("\nÚltimos valores:")
                # Eliminar filas con NaN para mostrar últimos valores válidos
                last_valid = df[custom_columns].dropna().tail(1)
                print(last_valid)
        except Exception as e:
            self.logger.error(f"Error verificando indicadores personalizados: {str(e)}")

def main():
    print("Iniciando pruebas de indicadores...")
    tester = TestIndicators()
    tester.test_all_indicators()

if __name__ == "__main__":
    main() 