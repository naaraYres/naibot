#!/usr/bin/env python3
"""
Bot de Trading Deriv - Punto de Entrada Principal
Versión modular con mejor organización y mantenibilidad.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Optional

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import TRADING_CONFIG
from config.credentials import get_credentials
from src.utils.logger import setup_logger
from src.trading_bot import TradingBot


def validate_environment() -> bool:
    """Valida que el entorno esté correctamente configurado."""
    logger = logging.getLogger(__name__)
    
    required_dirs = ['logs', 'data', 'config', 'src']
    missing_dirs = []
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)
    
    if missing_dirs:
        logger.error(f"Directorios faltantes: {missing_dirs}")
        logger.info("Creando directorios faltantes...")
        
        for directory in missing_dirs:
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Directorio creado: {directory}")
            except Exception as e:
                logger.error(f"Error creando directorio {directory}: {e}")
                return False
    
    return True


def load_trading_config() -> Optional[Dict]:
    """Carga y valida la configuración de trading."""
    logger = logging.getLogger(__name__)
    
    try:
        # Obtener credenciales
        credentials = get_credentials()
        if not credentials:
            logger.error("No se pudieron obtener las credenciales")
            return None
        
        # Combinar configuración base con credenciales
        config = {**TRADING_CONFIG, **credentials}
        
        # Validar parámetros críticos
        required_params = ['app_id', 'token', 'symbol', 'stake']
        missing_params = [p for p in required_params if not config.get(p)]
        
        if missing_params:
            logger.error(f"Parámetros faltantes en configuración: {missing_params}")
            return None
        
        logger.info("Configuración cargada correctamente")
        logger.info(f"Símbolo: {config['symbol']}")
        logger.info(f"Stake: ${config['stake']}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
        return None


def main():
    """Función principal del programa."""
    print("=" * 60)
    print("🤖 Bot de Trading Deriv - Versión Modular 2.0")
    print("=" * 60)
    
    try:
        # Configurar logging
        logger = setup_logger(
            name="main",
            log_file=f"logs/trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logger.info("Iniciando bot de trading...")
        
        # Validar entorno
        if not validate_environment():
            logger.error("Error en validación del entorno")
            return 1
        
        # Cargar configuración
        config = load_trading_config()
        if not config:
            logger.error("Error cargando configuración")
            return 1
        
        # Mostrar información de configuración
        logger.info("CONFIGURACIÓN ACTIVA:")
        logger.info(f"  App ID: {config['app_id']}")
        logger.info(f"  Símbolo: {config['symbol']}")
        logger.info(f"  Stake: ${config['stake']}")
        logger.info(f"  Duración: {config['duration']} {config['duration_unit']}")
        logger.info(f"  Granularidad: {config['granularity']}s")
        
        # Crear y ejecutar bot
        logger.info("Creando instancia del bot...")
        bot = TradingBot(config)
        
        logger.info("Iniciando ejecución del bot...")
        logger.info("Presiona Ctrl+C para detener el bot")
        
        # Ejecutar bot
        result = bot.run()
        
        if result:
            logger.info("Bot finalizado correctamente")
            return 0
        else:
            logger.error("Bot finalizado con errores")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido por el usuario")
        logger = logging.getLogger(__name__)
        logger.info("Bot detenido por el usuario (Ctrl+C)")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        logger = logging.getLogger(__name__)
        logger.error(f"Error crítico no manejado: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)