# IA LSTM ARIMA Trading Bot

Bot de trading que utiliza modelos LSTM y ARIMA para predecir tendencias en el mercado de criptomonedas.

## Estructura del Proyecto
```
project/
├── main.py              # Punto de entrada principal
├── bnc/                 # Integración con Binance
├── config/             # Configuraciones
├── data/               # Manejo de datos
├── indicators/         # Indicadores técnicos
├── tests/             # Pruebas
└── utils/             # Utilidades
```

## Instalación
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso
```bash
python main.py
```
