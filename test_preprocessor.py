from data.collector import DataCollector
from data.preprocessor import DataPreprocessor
import pandas as pd
pd.set_option('display.max_columns', None)

def test_preprocessing():
    # Obtener datos
    collector = DataCollector()
    preprocessor = DataPreprocessor()
    
    symbol = "BTCUSDT"
    timeframe = "1h"
    
    print(f"\nObteniendo y procesando datos para {symbol}")
    
    # Obtener datos históricos
    df = collector.get_historical_data(symbol, timeframe)
    
    if not df.empty:
        # Añadir indicadores técnicos
        print("\nAñadiendo indicadores técnicos...")
        df_with_indicators = preprocessor.add_technical_indicators(df)
        print("\nIndicadores técnicos añadidos:")
        print(df_with_indicators.columns.tolist())
        
        # Preparar datos para modelos
        print("\nPreparando datos para ARIMA y LSTM...")
        processed_data, scalers = preprocessor.prepare_data(df)
        
        # Mostrar información sobre los datos procesados
        print("\nForma de los datos LSTM:")
        print(f"X shape: {processed_data['lstm']['X'].shape}")
        print(f"y shape: {processed_data['lstm']['y'].shape}")
        
        print("\nPrimeras filas de retornos para ARIMA:")
        print(processed_data['arima']['returns'][:5])
        
    else:
        print("\n¡Error! No se pudieron obtener datos.")

if __name__ == "__main__":
    test_preprocessing() 