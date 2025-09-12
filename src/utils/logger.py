import logging
import os
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> None:
    """Configura el sistema de logging con formato mejorado."""
    log_format = '%(asctime)s [%(levelname)8s] %(name)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configurar nivel
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Crear directorio logs si no existe
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    
    # Archivo de log con timestamp
    log_filename = f"logs/trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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


def setup_logger(name: str = "trading_bot", log_file: str = None, level: int = logging.INFO):
    """
    Función wrapper para compatibilidad con el main.py.
    
    Args:
        name: Nombre del logger
        log_file: Ruta del archivo de log (se ignora, usa setup_logging)
        level: Nivel de logging
        
    Returns:
        Logger configurado
    """
    # Configurar logging si no está configurado
    if not logging.getLogger().handlers:
        log_level_name = logging.getLevelName(level)
        setup_logging(log_level_name)
    
    return logging.getLogger(name)


def get_logger(name: str = "trading_bot"):
    """Obtiene un logger ya configurado."""
    return logging.getLogger(name)