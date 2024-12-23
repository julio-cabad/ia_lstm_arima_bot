from data.collector import DataCollector
import pandas as pd
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas

def test_data_collection():
    collector = DataCollector()
    
    # Probar colección de datos históricos
    symbol = "BTCUSDT"
    timeframe = "1h"
    lookback_days = 7
    
    print(f"\nObteniendo datos históricos para {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Días hacia atrás: {lookback_days}")
    
    df = collector.get_historical_data(symbol, timeframe, lookback_days)
    
    if not df.empty:
        print("\nPrimeras 5 filas de datos:")
        print(df.head())
        print(f"\nColumnas disponibles: {df.columns.tolist()}")
        print(f"Total de registros: {len(df)}")
        print(f"\nEstadísticas básicas:")
        print(df.describe())
    else:
        print("\n¡Error! No se pudieron obtener datos.")

if __name__ == "__main__":
    test_data_collection() 