"""
Funciones auxiliares y utilidades para el trading bot.
Contiene herramientas comunes, validaciones y funciones de soporte.
"""

import re
import json
import hashlib
import logging
from datetime import datetime, timedelta
from decimal import Decimal, getcontext, InvalidOperation
from typing import Dict, List, Optional, Any, Union, Tuple
import time
import functools
from enum import Enum

# Configurar precisión decimal
getcontext().prec = 28


class ValidationError(Exception):
    """Error de validación personalizado."""
    pass


class TimeFrame(Enum):
    """Marcos temporales comunes."""
    M1 = 60        # 1 minuto
    M5 = 300       # 5 minutos
    M15 = 900      # 15 minutos
    M30 = 1800     # 30 minutos
    H1 = 3600      # 1 hora
    H4 = 14400     # 4 horas
    D1 = 86400     # 1 día


def safe_decimal(value: Any, default: Decimal = Decimal('0')) -> Decimal:
    """Convierte un valor a Decimal de forma segura."""
    try:
        if isinstance(value, Decimal):
            return value
        elif isinstance(value, (int, float)):
            return Decimal(str(value))
        elif isinstance(value, str):
            # Limpiar string (remover espacios, comas, etc.)
            cleaned = re.sub(r'[^\d.-]', '', value)
            return Decimal(cleaned) if cleaned else default
        else:
            return default
    except (InvalidOperation, ValueError):
        return default


def format_currency(amount: Union[Decimal, float, int], currency: str = "USD", 
                   decimals: int = 2) -> str:
    """Formatea un monto como moneda."""
    try:
        decimal_amount = safe_decimal(amount)
        formatted = f"{decimal_amount:.{decimals}f}"
        
        # Agregar separadores de miles
        parts = formatted.split('.')
        parts[0] = f"{int(parts[0]):,}"
        formatted = '.'.join(parts)
        
        return f"${formatted} {currency}" if currency else f"${formatted}"
        
    except Exception:
        return f"$0.00 {currency}"


def format_percentage(value: Union[Decimal, float, int], decimals: int = 2) -> str:
    """Formatea un valor como porcentaje."""
    try:
        decimal_value = safe_decimal(value)
        return f"{decimal_value:.{decimals}f}%"
    except Exception:
        return "0.00%"


def format_timestamp(timestamp: Union[datetime, float, int], 
                    format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Formatea un timestamp de forma consistente."""
    try:
        if isinstance(timestamp, datetime):
            dt = timestamp
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
        else:
            return str(timestamp)
        
        return dt.strftime(format_string)
        
    except Exception:
        return str(timestamp)


def validate_symbol(symbol: str) -> bool:
    """Valida que un símbolo tenga el formato correcto."""
    if not isinstance(symbol, str) or not symbol:
        return False
    
    # Patrones válidos para diferentes tipos de símbolos
    patterns = [
        r'^frx[A-Z]{6}$',          # Forex (ej: frxEURUSD)
        r'^R_\d+$',                # Índices sintéticos (ej: R_50, R_100)
        r'^crash\d+$',             # Crash indices
        r'^boom\d+$',              # Boom indices
        r'^step\d+$',              # Step indices
        r'^[A-Z]{3}[A-Z]{3}$'      # Pares de divisas tradicionales
    ]
    
    return any(re.match(pattern, symbol) for pattern in patterns)


def validate_stake(stake: Any, min_stake: Decimal = Decimal('0.35'), 
                  max_stake: Decimal = Decimal('2000')) -> Tuple[bool, str]:
    """Valida que el stake esté en el rango permitido."""
    try:
        stake_decimal = safe_decimal(stake)
        
        if stake_decimal < min_stake:
            return False, f"Stake mínimo: ${min_stake}"
        
        if stake_decimal > max_stake:
            return False, f"Stake máximo: ${max_stake}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Stake inválido: {e}"


def validate_duration(duration: int, duration_unit: str = "m") -> Tuple[bool, str]:
    """Valida la duración de un contrato."""
    valid_units = ["s", "m", "h", "d"]
    
    if duration_unit not in valid_units:
        return False, f"Unidad inválida. Usar: {valid_units}"
    
    # Límites por unidad
    limits = {
        "s": (15, 3600),      # 15 segundos a 1 hora
        "m": (1, 1440),       # 1 minuto a 24 horas
        "h": (1, 168),        # 1 hora a 7 días
        "d": (1, 365)         # 1 día a 1 año
    }
    
    min_val, max_val = limits.get(duration_unit, (1, 1440))
    
    if not (min_val <= duration <= max_val):
        return False, f"Duración debe estar entre {min_val} y {max_val} {duration_unit}"
    
    return True, ""


def calculate_win_rate(winning_trades: int, total_trades: int) -> Decimal:
    """Calcula la tasa de ganancia."""
    if total_trades == 0:
        return Decimal('0')
    
    return (Decimal(str(winning_trades)) / Decimal(str(total_trades))) * 100


def calculate_profit_factor(gross_profit: Decimal, gross_loss: Decimal) -> Decimal:
    """Calcula el factor de ganancia."""
    if gross_loss == 0:
        return Decimal('0') if gross_profit == 0 else Decimal('999')
    
    return gross_profit / gross_loss


def calculate_sharpe_ratio(returns: List[Decimal], risk_free_rate: Decimal = Decimal('0.02')) -> Decimal:
    """Calcula el ratio de Sharpe."""
    if len(returns) < 2:
        return Decimal('0')
    
    mean_return = sum(returns) / len(returns)
    
    # Calcular desviación estándar
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = variance.sqrt() if variance > 0 else Decimal('0')
    
    if std_dev == 0:
        return Decimal('0')
    
    return (mean_return - risk_free_rate) / std_dev


def calculate_max_drawdown(balance_history: List[Decimal]) -> Tuple[Decimal, Decimal]:
    """Calcula el máximo drawdown y su porcentaje."""
    if len(balance_history) < 2:
        return Decimal('0'), Decimal('0')
    
    peak = balance_history[0]
    max_drawdown = Decimal('0')
    max_drawdown_pct = Decimal('0')
    
    for balance in balance_history:
        if balance > peak:
            peak = balance
        
        drawdown = peak - balance
        drawdown_pct = (drawdown / peak) * 100 if peak > 0 else Decimal('0')
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_pct = drawdown_pct
    
    return max_drawdown, max_drawdown_pct


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, 
                    backoff_multiplier: float = 2.0):
    """Decorador para reintentar operaciones que fallan."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:  # No esperar en el último intento
                        time.sleep(current_delay)
                        current_delay *= backoff_multiplier
                    
                    logging.getLogger(func.__module__).warning(
                        f"Intento {attempt + 1}/{max_attempts} falló para {func.__name__}: {e}"
                    )
            
            # Si todos los intentos fallan, relanzar la última excepción
            raise last_exception
            
        return wrapper
    return decorator


def rate_limit(calls_per_second: float = 1.0):
    """Decorador para limitar la tasa de llamadas a una función."""
    min_interval = 1.0 / calls_per_second
    
    def decorator(func):
        last_called = [0.0]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
            
        return wrapper
    return decorator


def log_execution_time(func):
    """Decorador para medir tiempo de ejecución de funciones."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger = logging.getLogger(func.__module__)
        
        if execution_time > 1.0:  # Solo log si toma más de 1 segundo
            logger.warning(f"{func.__name__} tomó {execution_time:.2f}s en ejecutarse")
        else:
            logger.debug(f"{func.__name__} ejecutado en {execution_time:.3f}s")
        
        return result
    return wrapper


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitiza un nombre de archivo removiendo caracteres inválidos."""
    # Caracteres no permitidos en nombres de archivo
    invalid_chars = r'<>:"/\\|?*'
    
    # Reemplazar caracteres inválidos con guión bajo
    sanitized = re.sub(f'[{re.escape(invalid_chars)}]', '_', filename)
    
    # Remover espacios múltiples y al inicio/final
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Truncar si es muy largo
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def hash_string(text: str, algorithm: str = 'md5') -> str:
    """Genera un hash de un string."""
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(text.encode('utf-8'))
        return hash_obj.hexdigest()
    except Exception:
        return hashlib.md5(text.encode('utf-8')).hexdigest()


def parse_timeframe(timeframe_str: str) -> Optional[int]:
    """Convierte string de timeframe a segundos."""
    timeframe_str = timeframe_str.upper().strip()
    
    # Mapeo de timeframes comunes
    timeframes = {
        'M1': 60, '1M': 60, '1MIN': 60,
        'M5': 300, '5M': 300, '5MIN': 300,
        'M15': 900, '15M': 900, '15MIN': 900,
        'M30': 1800, '30M': 1800, '30MIN': 1800,
        'H1': 3600, '1H': 3600, '1HOUR': 3600,
        'H4': 14400, '4H': 14400, '4HOUR': 14400,
        'D1': 86400, '1D': 86400, '1DAY': 86400, 'DAILY': 86400
    }
    
    return timeframes.get(timeframe_str)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Divide una lista en chunks de tamaño específico."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Aplana un diccionario anidado."""
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Fusiona dos diccionarios de forma recursiva."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def get_memory_usage() -> Dict[str, float]:
    """Obtiene información de uso de memoria (si psutil está disponible)."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'error': 'psutil no está disponible'}
    except Exception as e:
        return {'error': str(e)}


def is_market_open(symbol: str, current_time: Optional[datetime] = None) -> Tuple[bool, str]:
    """Verifica si el mercado está abierto para un símbolo dado."""
    if current_time is None:
        current_time = datetime.now()
    
    # Para índices sintéticos, siempre están abiertos
    if symbol.startswith('R_') or 'crash' in symbol.lower() or 'boom' in symbol.lower():
        return True, "Mercado sintético - siempre abierto"
    
    # Para forex, verificar horarios (simplificado)
    if symbol.startswith('frx'):
        weekday = current_time.weekday()  # 0=Lunes, 6=Domingo
        hour = current_time.hour
        
        # Forex cerrado los fines de semana
        if weekday == 5 and hour >= 21:  # Viernes después de 21:00
            return False, "Mercado forex cerrado - fin de semana"
        elif weekday == 6:  # Todo el sábado
            return False, "Mercado forex cerrado - fin de semana"
        elif weekday == 0 and hour < 1:  # Domingo antes de 01:00
            return False, "Mercado forex cerrado - fin de semana"
        
        return True, "Mercado forex abierto"
    
    # Para otros símbolos, asumir abierto
    return True, "Horario no verificado"


def calculate_position_correlation(positions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calcula la correlación entre posiciones abiertas."""
    if len(positions) < 2:
        return {}
    
    # Simplificado - en una implementación real usarías datos históricos
    correlations = {}
    
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions[i+1:], i+1):
            pair_key = f"{pos1.get('symbol', 'N/A')}-{pos2.get('symbol', 'N/A')}"
            
            # Correlación simulada basada en tipos de símbolos
            symbol1 = pos1.get('symbol', '').lower()
            symbol2 = pos2.get('symbol', '').lower()
            
            if symbol1 == symbol2:
                correlation = 1.0
            elif 'eur' in symbol1 and 'eur' in symbol2:
                correlation = 0.7
            elif 'usd' in symbol1 and 'usd' in symbol2:
                correlation = 0.6
            elif symbol1.startswith('r_') and symbol2.startswith('r_'):
                correlation = 0.3
            else:
                correlation = 0.1
            
            correlations[pair_key] = correlation
    
    return correlations


def generate_trade_id(symbol: str, timestamp: Optional[datetime] = None) -> str:
    """Genera un ID único para un trade."""
    if timestamp is None:
        timestamp = datetime.now()
    
    # Formato: SYMBOL_YYYYMMDD_HHMMSS_HASH
    date_str = timestamp.strftime('%Y%m%d_%H%M%S')
    hash_input = f"{symbol}_{timestamp.timestamp()}"
    short_hash = hash_string(hash_input)[:8].upper()
    
    return f"{symbol}_{date_str}_{short_hash}"


def parse_contract_type(contract_type: str) -> Tuple[str, str]:
    """Parsea el tipo de contrato y retorna tipo normalizado y descripción."""
    contract_type = contract_type.upper().strip()
    
    contract_mapping = {
        'CALL': ('CALL', 'Higher/Up'),
        'PUT': ('PUT', 'Lower/Down'),
        'CALLE': ('CALL', 'Rise/Higher'),
        'PUTE': ('PUT', 'Fall/Lower'),
        'HIGHER': ('CALL', 'Higher'),
        'LOWER': ('PUT', 'Lower'),
        'UP': ('CALL', 'Up'),
        'DOWN': ('PUT', 'Down'),
        'RISE': ('CALL', 'Rise'),
        'FALL': ('PUT', 'Fall')
    }
    
    return contract_mapping.get(contract_type, (contract_type, contract_type))


def calculate_risk_metrics(trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula métricas de riesgo completas."""
    if not trade_history:
        return {}
    
    profits = [safe_decimal(trade.get('profit', 0)) for trade in trade_history]
    
    # Métricas básicas
    total_trades = len(trade_history)
    winning_trades = sum(1 for p in profits if p > 0)
    losing_trades = total_trades - winning_trades
    
    total_profit = sum(profits)
    gross_profit = sum(p for p in profits if p > 0)
    gross_loss = abs(sum(p for p in profits if p < 0))
    
    avg_win = gross_profit / winning_trades if winning_trades > 0 else Decimal('0')
    avg_loss = gross_loss / losing_trades if losing_trades > 0 else Decimal('0')
    
    # Métricas avanzadas
    win_rate = calculate_win_rate(winning_trades, total_trades)
    profit_factor = calculate_profit_factor(gross_profit, gross_loss)
    
    # Calcular rachas
    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    
    for profit in profits:
        if profit > 0:
            current_win_streak += 1
            current_loss_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
        else:
            current_loss_streak += 1
            current_win_streak = 0
            max_loss_streak = max(max_loss_streak, current_loss_streak)
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': float(win_rate),
        'total_profit': float(total_profit),
        'gross_profit': float(gross_profit),
        'gross_loss': float(gross_loss),
        'profit_factor': float(profit_factor),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'risk_reward_ratio': float(avg_win / avg_loss) if avg_loss > 0 else 0
    }


def format_duration(seconds: int) -> str:
    """Formatea una duración en segundos a string legible."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds}s" if remaining_seconds else f"{minutes}m"
    elif seconds < 86400:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{hours}h {remaining_minutes}m" if remaining_minutes else f"{hours}h"
    else:
        days = seconds // 86400
        remaining_hours = (seconds % 86400) // 3600
        return f"{days}d {remaining_hours}h" if remaining_hours else f"{days}d"


def create_backup_filename(prefix: str = "backup", extension: str = "json") -> str:
    """Crea un nombre de archivo de backup con timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}.{extension}"


def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
    """Valida que un JSON tenga los campos requeridos."""
    missing_fields = []
    
    for field in required_fields:
        if '.' in field:  # Campo anidado
            parts = field.split('.')
            current = data
            
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    missing_fields.append(field)
                    break
        else:
            if field not in data:
                missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields


def safe_divide(numerator: Union[Decimal, float, int], 
                denominator: Union[Decimal, float, int], 
                default: Decimal = Decimal('0')) -> Decimal:
    """División segura que evita división por cero."""
    try:
        num = safe_decimal(numerator)
        den = safe_decimal(denominator)
        
        if den == 0:
            return default
        
        return num / den
        
    except Exception:
        return default


def get_timeframe_info(granularity_seconds: int) -> Dict[str, Any]:
    """Obtiene información sobre un timeframe basado en su granularidad."""
    timeframe_info = {
        60: {'name': '1M', 'display': '1 Minuto', 'bars_per_hour': 60, 'bars_per_day': 1440},
        300: {'name': '5M', 'display': '5 Minutos', 'bars_per_hour': 12, 'bars_per_day': 288},
        900: {'name': '15M', 'display': '15 Minutos', 'bars_per_hour': 4, 'bars_per_day': 96},
        1800: {'name': '30M', 'display': '30 Minutos', 'bars_per_hour': 2, 'bars_per_day': 48},
        3600: {'name': '1H', 'display': '1 Hora', 'bars_per_hour': 1, 'bars_per_day': 24},
        14400: {'name': '4H', 'display': '4 Horas', 'bars_per_hour': 0.25, 'bars_per_day': 6},
        86400: {'name': '1D', 'display': '1 Día', 'bars_per_hour': 1/24, 'bars_per_day': 1}
    }
    
    return timeframe_info.get(granularity_seconds, {
        'name': f'{granularity_seconds}s',
        'display': f'{granularity_seconds} Segundos',
        'bars_per_hour': 3600 / granularity_seconds,
        'bars_per_day': 86400 / granularity_seconds
    })


def compress_data(data: Union[str, bytes], method: str = 'gzip') -> bytes:
    """Comprime datos usando el método especificado."""
    try:
        import gzip
        import zlib
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if method == 'gzip':
            return gzip.compress(data)
        elif method == 'zlib':
            return zlib.compress(data)
        else:
            return data
            
    except ImportError:
        return data if isinstance(data, bytes) else data.encode('utf-8')


def decompress_data(compressed_data: bytes, method: str = 'gzip') -> bytes:
    """Descomprime datos usando el método especificado."""
    try:
        import gzip
        import zlib
        
        if method == 'gzip':
            return gzip.decompress(compressed_data)
        elif method == 'zlib':
            return zlib.decompress(compressed_data)
        else:
            return compressed_data
            
    except ImportError:
        return compressed_data


def generate_summary_stats(data: List[Union[int, float, Decimal]]) -> Dict[str, float]:
    """Genera estadísticas resumen de una lista de números."""
    if not data:
        return {}
    
    decimal_data = [safe_decimal(x) for x in data]
    n = len(decimal_data)
    
    # Estadísticas básicas
    total = sum(decimal_data)
    mean = total / n
    
    # Mediana
    sorted_data = sorted(decimal_data)
    if n % 2 == 0:
        median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        median = sorted_data[n//2]
    
    # Varianza y desviación estándar
    variance = sum((x - mean) ** 2 for x in decimal_data) / (n - 1) if n > 1 else Decimal('0')
    std_dev = variance.sqrt() if variance > 0 else Decimal('0')
    
    return {
        'count': n,
        'sum': float(total),
        'mean': float(mean),
        'median': float(median),
        'min': float(min(decimal_data)),
        'max': float(max(decimal_data)),
        'std_dev': float(std_dev),
        'variance': float(variance)
    }


class ConfigValidator:
    """Validador de configuraciones con reglas personalizables."""
    
    def __init__(self):
        self.rules = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_rule(self, field_path: str, validation_func, error_message: str):
        """Agrega una regla de validación."""
        self.rules[field_path] = {
            'validator': validation_func,
            'message': error_message
        }
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida una configuración contra todas las reglas."""
        errors = []
        
        for field_path, rule in self.rules.items():
            try:
                value = self._get_nested_value(config, field_path)
                
                if not rule['validator'](value):
                    errors.append(f"{field_path}: {rule['message']}")
                    
            except KeyError:
                errors.append(f"{field_path}: Campo requerido faltante")
            except Exception as e:
                errors.append(f"{field_path}: Error de validación - {e}")
        
        return len(errors) == 0, errors
    
    def _get_nested_value(self, data: Dict[str, Any], path: str):
        """Obtiene un valor anidado usando notación de punto."""
        parts = path.split('.')
        current = data
        
        for part in parts:
            current = current[part]
        
        return current


# Ejemplo de uso de funciones auxiliares
if __name__ == "__main__":
    # Configurar logging para pruebas
    logging.basicConfig(level=logging.INFO)
    
    # Probar conversiones
    print("=== Pruebas de conversiones ===")
    print(f"safe_decimal('123.45'): {safe_decimal('123.45')}")
    print(f"format_currency(1234.56): {format_currency(1234.56)}")
    print(f"format_percentage(0.1234): {format_percentage(0.1234)}")
    
    # Probar validaciones
    print("\n=== Pruebas de validaciones ===")
    print(f"validate_symbol('frxEURUSD'): {validate_symbol('frxEURUSD')}")
    print(f"validate_stake(5.0): {validate_stake(5.0)}")
    
    # Probar métricas
    print("\n=== Pruebas de métricas ===")
    sample_trades = [
        {'profit': 10}, {'profit': -5}, {'profit': 15}, {'profit': -8}, {'profit': 12}
    ]
    metrics = calculate_risk_metrics(sample_trades)
    print(f"Risk metrics: {json.dumps(metrics, indent=2)}")
    
    # Probar utilidades
    print("\n=== Pruebas de utilidades ===")
    print(f"generate_trade_id('EURUSD'): {generate_trade_id('EURUSD')}")
    print(f"format_duration(3661): {format_duration(3661)}")
    print(f"get_timeframe_info(300): {get_timeframe_info(300)}")