import time
from typing import Dict, List, Tuple
from datetime import datetime
from config.setting import CRYPTOCURRENCY_LIST
from utils.logger import setup_logger
from bnc.binance import RobotBinance
from data.collector import DataCollector
from data.preprocessor import DataPreprocessor
from models.lstm_model import LSTMPredictor
from models.arima_model import ArimaPredictor

class IAAgentBot:
    def __init__(self, timeframe: str):
        self.logger = setup_logger("IAAgentBot")
        self.timeframe = timeframe
        self.symbols = CRYPTOCURRENCY_LIST
        self.clients: Dict[str, RobotBinance] = {}
        self.data_collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.lstm_predictor = None
        self.arima_predictor = None
        self.initialize_clients()
    
    def initialize_clients(self):
        
        """Initialize Binance clients for each symbol."""
        self.logger.info("Initializing trading clients...")
        initialized_count = 0
        for symbol in self.symbols:
            try:
                client = RobotBinance(pair=symbol, temporality=self.timeframe)
                price = client.symbol_price()
                
                if price and price > 0:
                    self.clients[symbol] = client
                    self.logger.debug(f"Initialized client for {symbol}")
                    initialized_count += 1
                else:
                    self.logger.warning(f"Could not get valid price for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize client for {symbol}: {str(e)}")

    def initialize_lstm(self, input_shape: Tuple[int, int]):
        self.lstm_predictor = LSTMPredictor(input_shape)

    def initialize_arima(self):
        self.arima_predictor = ArimaPredictor()

    def analyze_market(self, symbol: str) -> Dict:
        """Analiza el mercado usando ARIMA y LSTM"""
        try:
            # Obtener datos históricos
            historical_data = self.data_collector.get_historical_data(
                symbol, 
                self.timeframe
            )
            
            # Preprocesar datos para LSTM
            X, y, scaler = self.preprocessor.prepare_data_for_lstm(historical_data)
            
            # Inicializar y entrenar LSTM
            if not self.lstm_predictor:
                self.initialize_lstm((X.shape[1], 1))
            self.lstm_predictor.train(X, y)
            
            # Predecir con LSTM
            lstm_prediction = self.lstm_predictor.predict(X[-1].reshape(1, X.shape[1], 1))
            lstm_prediction = scaler.inverse_transform(lstm_prediction)
            
            return lstm_prediction
            
        except Exception as e:
            self.logger.error(f"Error analyzing market for {symbol}: {str(e)}")
            return None

    def combine_predictions(self, arima_pred, lstm_pred) -> Dict:
        """Combina las predicciones de ARIMA y LSTM"""
        # Implementar lógica de combinación
        pass

    def get_signal(self, data):
        lstm_signal = self.lstm_predictor.predict(data)
        arima_signal = self.arima_predictor.predict(data)
        
        # Tomar decisión basada en ambos modelos
        if lstm_signal > 0 and arima_signal > 0:
            return "BUY"
        elif lstm_signal < 0 and arima_signal < 0:
            return "SELL"
        return "HOLD"

    def run(self):
        """Ejecuta el bot de trading"""
        while True:
            try:
                print("\033[2J\033[H")  # Limpiar pantalla
                print(f"=== TrendMagic Bot Status === {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Analizar cada símbolo
                for symbol in self.clients:
                    prediction = self.analyze_market(symbol)
                    if prediction:
                        # Implementar lógica de trading basada en predicciones
                        pass
                        
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Fatal error: {str(e)}")
                break
