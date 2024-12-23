import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
from utils.logger import setup_logger
from indicators.indicators import Indicators # Importamos la función que añade todos los indicadores

class DataPreprocessor:
    def __init__(self):
        self.logger = setup_logger("DataPreprocessor")
        self.scalers: Dict[str, MinMaxScaler] = {}
        
    def prepare_data(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[Dict[str, np.ndarray], Dict[str, MinMaxScaler]]:
        """
        Prepara los datos para los modelos ARIMA y LSTM.
        
        Args:
            df: DataFrame con datos históricos
            sequence_length: Longitud de la secuencia para LSTM
            
        Returns:
            Tuple con diccionarios conteniendo datos procesados y scalers
        """
        try:
            # Verificar datos
            if df.empty:
                raise ValueError("DataFrame está vacío")
                
            # Crear copia para no modificar los datos originales
            data = df.copy()
            
            # Asegurar que el índice está ordenado
            data = data.sort_index()
            
            # Añadir indicadores técnicos
            data = add_all_indicators(data)
            
            # Calcular retornos logarítmicos para ARIMA
            data['returns'] = np.log(data['close'] / data['close'].shift(1))
            data = data.dropna()
            
            # Preparar datos para ARIMA
            arima_data = {
                'returns': data['returns'].values,
                'close': data['close'].values
            }
            
            # Preparar datos para LSTM
            lstm_data = self._prepare_lstm_data(data, sequence_length)
            
            processed_data = {
                'arima': arima_data,
                'lstm': lstm_data
            }
            
            return processed_data, self.scalers
            
        except Exception as e:
            self.logger.error(f"Error en el preprocesamiento de datos: {str(e)}")
            raise
            
    def _prepare_lstm_data(self, data: pd.DataFrame, sequence_length: int) -> Dict[str, np.ndarray]:
        """
        Prepara los datos específicamente para LSTM.
        """
        try:
            # Seleccionar características para LSTM (incluir indicadores técnicos)
            features = ['open', 'high', 'low', 'close', 'volume', 
                       'RSX', 'EMA', 'BB_upper', 'BB_middle', 'BB_lower']  # Ajustar según tus indicadores
            
            # Asegurarse de que todas las características existen
            available_features = [f for f in features if f in data.columns]
            
            # Normalizar datos
            self.scalers['lstm'] = MinMaxScaler(feature_range=(0, 1))
            scaled_data = self.scalers['lstm'].fit_transform(data[available_features])
            
            # Crear secuencias para LSTM
            X, y = [], []
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:(i + sequence_length)])
                y.append(scaled_data[i + sequence_length, available_features.index('close')])
                
            return {
                'X': np.array(X),
                'y': np.array(y),
                'features': available_features
            }
            
        except Exception as e:
            self.logger.error(f"Error preparando datos para LSTM: {str(e)}")
            raise