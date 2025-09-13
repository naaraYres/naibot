import os
from decimal import Decimal
from typing import Dict, Any

# ==============================================================================
#                      CONFIGURACIÓN GENERAL DEL BOT
# ==============================================================================
# Define los parámetros de trading que se aplican a todos los símbolos
# Los valores se leen de variables de entorno o usan los predeterminados.
TRADING_CONFIG: Dict[str, Any] = {
    "stake": os.getenv("DERIV_STAKE", "1"),
    "duration": os.getenv("DERIV_DURATION", "5"),
    "duration_unit": os.getenv("DERIV_DURATION_UNIT", "m"),
    "granularity": os.getenv("DERIV_GRANULARITY", "60"),
    "max_daily_loss": os.getenv("DERIV_MAX_DAILY_LOSS", "10"),
    "max_consecutive_losses": os.getenv("DERIV_MAX_CONSECUTIVE_LOSSES", "3"),
    "tolerancia_zona": os.getenv("DERIV_ZONE_TOLERANCE", "0.0002"),
    "ema_fast_period": int(os.getenv("DERIV_EMA_FAST", "8")),
    "ema_slow_period": int(os.getenv("DERIV_EMA_SLOW", "21")),
    
    # Parámetros de estrategia globales por defecto
    "default_strategy_params": {
        "pattern": os.getenv("DERIV_PATTERN", "M"),
        "risk": float(os.getenv("DERIV_RISK", "1.0")),
        "confirmation": int(os.getenv("DERIV_CONFIRMATION", "2")),
        "min_pattern_height": float(os.getenv("DERIV_MIN_PATTERN_HEIGHT", "0.001")),
        "max_pattern_width": int(os.getenv("DERIV_MAX_PATTERN_WIDTH", "20"))
    }
}

# ==============================================================================
#               CONFIGURACIÓN DE ESTRATEGIA POR SÍMBOLO
# ==============================================================================
# Parámetros específicos para la estrategia de confluencia, ajustados por símbolo.
SYMBOL_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "frxEURUSD": {
        "MONTHLY_HIGH": Decimal('1.20000'),
        "MONTHLY_LOW": Decimal('1.10000'),
        "SUPPORTS": [Decimal('1.12000'), Decimal('1.13000'), Decimal('1.14000'), Decimal('1.15000')],
        "RESISTANCES": [Decimal('1.18000'), Decimal('1.17000'), Decimal('1.16000')],
        "MID_LEVEL": Decimal('1.15000'),
        "strategy_params": {
            "pattern": "M",
            "risk": 1.5,
            "confirmation": 2,
            "min_pattern_height": 0.0015,
            "max_pattern_width": 15
        }
    },
    "R_50": {
        "MONTHLY_HIGH": Decimal('165000'),
        "MONTHLY_LOW": Decimal('135000'),
        "SUPPORTS": [Decimal('140000'), Decimal('145000'), Decimal('150000')],
        "RESISTANCES": [Decimal('160000'), Decimal('155000')],
        "MID_LEVEL": Decimal('150000'),
        "strategy_params": {
            "pattern": "W",
            "risk": 1.0,
            "confirmation": 3,
            "min_pattern_height": 2000,
            "max_pattern_width": 25
        }
    },
    "R_100": {
        "MONTHLY_HIGH": Decimal('330000'),
        "MONTHLY_LOW": Decimal('270000'),
        "SUPPORTS": [Decimal('280000'), Decimal('290000'), Decimal('300000')],
        "RESISTANCES": [Decimal('320000'), Decimal('310000')],
        "MID_LEVEL": Decimal('300000'),
        "strategy_params": {
            "pattern": "HCH",
            "risk": 0.8,
            "confirmation": 2,
            "min_pattern_height": 4000,
            "max_pattern_width": 20
        }
    }
}

# ==============================================================================
#                    FUNCIONES AUXILIARES DE CONFIGURACIÓN
# ==============================================================================

def get_strategy_params(symbol: str) -> Dict[str, Any]:
    """
    Obtiene los parámetros de estrategia para un símbolo específico.
    
    Args:
        symbol: El símbolo del activo (ej: "frxEURUSD", "R_50")
        
    Returns:
        Dict con los parámetros de estrategia
    """
    if symbol in SYMBOL_STRATEGIES and "strategy_params" in SYMBOL_STRATEGIES[symbol]:
        return SYMBOL_STRATEGIES[symbol]["strategy_params"]
    
    # Si no existe configuración específica, usar parámetros por defecto
    return TRADING_CONFIG["default_strategy_params"]

def get_symbol_levels(symbol: str) -> Dict[str, Any]:
    """
    Obtiene los niveles de soporte/resistencia para un símbolo específico.
    
    Args:
        symbol: El símbolo del activo
        
    Returns:
        Dict con los niveles del símbolo
    """
    if symbol not in SYMBOL_STRATEGIES:
        raise ValueError(f"Símbolo {symbol} no encontrado en SYMBOL_STRATEGIES")
    
    return {key: value for key, value in SYMBOL_STRATEGIES[symbol].items() 
            if key != "strategy_params"}

def build_bot_config(app_id: str, token: str, symbol: str) -> Dict[str, Any]:
    """
    Construye la configuración completa para el TradingBot.
    Esta función resuelve el acceso a strategy_params que espera el bot.
    
    Args:
        app_id: ID de la aplicación Deriv
        token: Token de autorización
        symbol: Símbolo del activo a operar
        
    Returns:
        Dict con toda la configuración necesaria para el bot
        
    Raises:
        ValueError: Si el símbolo no está configurado
    """
    # Verificar que el símbolo existe
    if symbol not in SYMBOL_STRATEGIES:
        available_symbols = list(SYMBOL_STRATEGIES.keys())
        raise ValueError(f"Símbolo '{symbol}' no encontrado. Símbolos disponibles: {available_symbols}")
    
    # Configuración base del trading
    config = TRADING_CONFIG.copy()
    
    # Agregar parámetros de conexión
    config.update({
        'app_id': app_id,
        'token': token,
        'symbol': symbol
    })
    
    # Crear el diccionario strategy_params que espera el TradingBot
    # El bot accede como: config['strategy_params'][config['symbol']]
    config['strategy_params'] = {}
    
    # Llenar strategy_params para todos los símbolos disponibles
    for sym in SYMBOL_STRATEGIES.keys():
        config['strategy_params'][sym] = get_strategy_params(sym)
    
    return config

def get_available_symbols() -> list:
    """
    Obtiene la lista de símbolos disponibles para trading.
    
    Returns:
        Lista de símbolos configurados
    """
    return list(SYMBOL_STRATEGIES.keys())

def validate_symbol(symbol: str) -> bool:
    """
    Valida si un símbolo está configurado.
    
    Args:
        symbol: Símbolo a validar
        
    Returns:
        True si el símbolo está configurado, False caso contrario
    """
    return symbol in SYMBOL_STRATEGIES

def print_config_summary(symbol: str = None) -> None:
    """
    Imprime un resumen de la configuración.
    
    Args:
        symbol: Símbolo específico a mostrar. Si es None, muestra todos.
    """
    print("=" * 60)
    print("RESUMEN DE CONFIGURACIÓN DEL BOT")
    print("=" * 60)
    
    # Configuración general
    print("\n📊 CONFIGURACIÓN GENERAL:")
    print(f"   Stake por operación: ${TRADING_CONFIG['stake']}")
    print(f"   Duración: {TRADING_CONFIG['duration']}{TRADING_CONFIG['duration_unit']}")
    print(f"   Granularidad: {TRADING_CONFIG['granularity']}s")
    print(f"   Límite pérdida diaria: ${TRADING_CONFIG['max_daily_loss']}")
    print(f"   Límite pérdidas consecutivas: {TRADING_CONFIG['max_consecutive_losses']}")
    print(f"   Tolerancia de zona: {TRADING_CONFIG['tolerancia_zona']}")
    print(f"   EMAs: {TRADING_CONFIG['ema_fast_period']}/{TRADING_CONFIG['ema_slow_period']}")
    
    # Símbolos disponibles
    symbols_to_show = [symbol] if symbol and validate_symbol(symbol) else get_available_symbols()
    
    print(f"\n🎯 SÍMBOLOS CONFIGURADOS ({len(symbols_to_show)}):")
    
    for sym in symbols_to_show:
        strategy_params = get_strategy_params(sym)
        levels = get_symbol_levels(sym)
        
        print(f"\n   📈 {sym}:")
        print(f"      Patrón: {strategy_params['pattern']}")
        print(f"      Riesgo: {strategy_params['risk']}%")
        print(f"      Confirmación: {strategy_params['confirmation']} velas")
        print(f"      Rango mensual: {levels['MONTHLY_LOW']} - {levels['MONTHLY_HIGH']}")
        print(f"      Soportes: {len(levels['SUPPORTS'])} niveles")
        print(f"      Resistencias: {len(levels['RESISTANCES'])} niveles")
    
    print("=" * 60)

# ==============================================================================
#                           EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    # Ejemplo de uso de las funciones
    print_config_summary()
    
    # Ejemplo de construcción de configuración para el bot
    try:
        config = build_bot_config(
            app_id="12345", 
            token="tu_token_aqui", 
            symbol="frxEURUSD"
        )
        print(f"\n✅ Configuración construida para frxEURUSD")
        print(f"Strategy params: {config['strategy_params']['frxEURUSD']}")
        
    except ValueError as e:
        print(f"❌ Error: {e}")