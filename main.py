from trading.bot import IAAgentBot
from utils.logger import setup_logger


def main():
    """Punto de entrada principal del bot"""
    print("Starting TrendMagic Bot")
    logger = setup_logger("Main")
    
    try:  
        bot = IAAgentBot(timeframe="1h")
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()