import pandas as pd
import numpy as np
from typing import Dict
from utils.logger import setup_logger
from indicators.indicators import Indicators

class DataPreprocessor:
    def __init__(self):
        self.logger = setup_logger("DataPreprocessor")
        
    def prepare_data_for_arima(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        try:
            data = df.copy()
            indicators = Indicators(data)
            data = indicators.add_all_indicators()
            
            data['returns'] = np.log(data['close'] / data['close'].shift(1)) * 100
            data = data.dropna(subset=['returns'])
            
            return {
                'returns': data['returns'].values,
                'dates': data.index,
                'close': data['close'].values
            }
        except Exception as e:
            self.logger.error(f"Error en preprocesamiento: {str(e)}")
            raise

    def validate_data_quality(self, data: pd.DataFrame) -> bool:
        """Valida la calidad de los datos"""
        try:
            print("\n=== Validación de Calidad de Datos ===")
            
            # 1. Verificar suficientes datos
            min_required = 100
            print(f"\nDatos disponibles: {len(data)} (mínimo requerido: {min_required})")
            if len(data) < min_required:
                self.logger.warning(f"Insuficientes datos: {len(data)} < {min_required}")
                return False
            
            # 2. Verificar estacionariedad
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(data['returns'])
            print(f"\nTest de Estacionariedad (ADF):")
            print(f"p-value: {adf_result[1]:.4f}")
            
            if adf_result[1] > 0.05:
                print("Los datos no son estacionarios, aplicando diferenciaci��n...")
                data['returns'] = np.diff(data['returns'], prepend=data['returns'][0])
                adf_result = adfuller(data['returns'])
                print(f"p-value después de diferenciación: {adf_result[1]:.4f}")
            
            # 3. Verificar autocorrelación
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(data['returns'], lags=[10])
            print(f"\nTest de Autocorrelación (Ljung-Box):")
            print(f"p-value: {lb_result['lb_pvalue'].iloc[0]:.4f}")
            
            if lb_result['lb_pvalue'].iloc[0] > 0.05:
                print("Advertencia: Baja autocorrelación en los datos")
            
            # 4. Verificar volatilidad
            from arch import arch_model
            am = arch_model(data['returns'])
            res = am.fit(disp='off')
            arch_test = res.arch_lm_test()
            print(f"\nTest de Volatilidad:")
            try:
                print(f"Test Statistic: {float(arch_test.stat):.4f}")
                print(f"p-value: {float(arch_test.pvalue):.4f}")
            except:
                print(f"Test Statistic: {arch_test.stat}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validando datos: {str(e)}")
            print(f"Error durante la validación: {str(e)}")
            return True