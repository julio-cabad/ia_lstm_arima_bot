def test_trading_strategy():
    bot = TradingBot()
    results = bot.backtest(
        symbol="BTCUSDT",
        timeframe="1h",
        initial_balance=1000,
        start_date="2024-01-01"
    ) 