import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data.collector import DataCollector
from indicators.indicators import Indicators
from utils.logger import setup_logger

class TestIndicators:
    def __init__(self):
        self.logger = setup_logger("TestIndicators")
        self.collector = DataCollector()
        
    def test_indicators(self):
        """Prueba los indicadores para verificar dimensiones y valores nulos"""
        try:
            print("\nIniciando prueba de indicadores...")
            
            # Obtener datos
            symbol = "BTCUSDT"
            timeframe = "1h"
            df = self.collector.get_historical_data(symbol, timeframe)
            
            if df.empty:
                self.logger.error("No se pudieron obtener datos para las pruebas")
                return
            
            # Calcular indicadores
            indicators = Indicators(df)
            data_with_indicators = indicators.add_all_indicators()
            
            # Verificar dimensiones
            print(f"Dimensiones de los datos con indicadores: {data_with_indicators.shape}")
            
            # Verificar valores nulos
            null_counts = data_with_indicators.isnull().sum()
            print("\nValores nulos por columna:")
            print(null_counts)
            
            if null_counts.any():
                raise ValueError("Existen valores nulos en los indicadores")
            
            print("\nPrueba de indicadores completada exitosamente")
            
        except Exception as e:
            self.logger.error(f"Error en prueba de indicadores: {str(e)}")
            raise

def main():
    tester = TestIndicators()
    tester.test_indicators()

if __name__ == "__main__":
    main() 