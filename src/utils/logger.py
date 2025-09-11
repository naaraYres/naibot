import logging
from datetime import datetime

def setup_logging(log_level: str = "INFO") -> None:
    """Configura el sistema de logging con formato mejorado."""
    log_format = '%(asctime)s [%(levelname)8s] %(name)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configurar nivel
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Archivo de log con timestamp
    log_filename = f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Configurar loggers específicos
    logging.getLogger('websocket').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    print(f"📝 Logs guardándose en: {log_filename}")