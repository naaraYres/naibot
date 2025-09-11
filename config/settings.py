import os
from decimal import Decimal
from typing import Dict

def get_trading_config() -> Dict:
    """Carga configuración desde variables de entorno con valores por defecto."""
    return {
        "stake": os.getenv("DERIV_STAKE", "1"),
        "duration": os.getenv("DERIV_DURATION", "5"),
        "duration_unit": os.getenv("DERIV_DURATION_UNIT", "m"),
        "granularity": os.getenv("DERIV_GRANULARITY", "60"),
        "max_daily_loss": os.getenv("DERIV_MAX_DAILY_LOSS", "10"),
        "max_consecutive_losses": os.getenv("DERIV_MAX_CONSECUTIVE_LOSSES", "3"),
        "tolerancia_zona": os.getenv("DERIV_ZONE_TOLERANCE", "0.0002"),
        "ema_fast_period": int(os.getenv("DERIV_EMA_FAST", "8")),
        "ema_slow_period": int(os.getenv("DERIV_EMA_SLOW", "21")),
    }

def get_symbol_strategies() -> Dict:
    """Estrategias por símbolo."""
    return {
        "frxEURUSD": {
            "MONTHLY_HIGH": Decimal('1.20000'),
            "MONTHLY_LOW": Decimal('1.10000'),
            "SUPPORTS": [Decimal('1.12000'), Decimal('1.13000'), Decimal('1.14000'), Decimal('1.15000')],
            "RESISTANCES": [Decimal('1.18000'), Decimal('1.17000'), Decimal('1.16000')],
            "MID_LEVEL": Decimal('1.15000')
        },
        "R_50": {
            "MONTHLY_HIGH": Decimal('165000'),
            "MONTHLY_LOW": Decimal('135000'),
            "SUPPORTS": [Decimal('140000'), Decimal('145000'), Decimal('150000')],
            "RESISTANCES": [Decimal('160000'), Decimal('155000')],
            "MID_LEVEL": Decimal('150000')
        },
        "R_100": {
            "MONTHLY_HIGH": Decimal('330000'),
            "MONTHLY_LOW": Decimal('270000'),
            "SUPPORTS": [Decimal('280000'), Decimal('290000'), Decimal('300000')],
            "RESISTANCES": [Decimal('320000'), Decimal('310000')],
            "MID_LEVEL": Decimal('300000')
        }
    }