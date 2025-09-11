"""
Manejo de datos de mercado y conexión con fuentes de datos.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from threading import Lock

class MarketDataBuffer:
    """Buffer optimizado para datos de mercado con thread safety."""
    
    def __init__(self, max_size: int = 1000):
        """
        Inicializa el buffer de datos de mercado.
        
        Args:
            max_size: Tamaño máximo del buffer
        """
        self.logger = logging.getLogger("MarketDataBuffer")
        self.max_size = max_size
        self.lock = Lock()
        
        # Buffers para diferentes tipos de datos
        self.candles = deque(maxlen=max_size)
        self.ticks = deque(maxlen=max_size * 10)  # Más ticks que velas
        
        # Metadatos
        self.last_update = None
        self.symbol = None
        self.granularity = None
        
        # Estadísticas
        self.total_candles_received = 0
        self.total_ticks_received = 0
        self.data_gaps_detected = 0
    
    def add_candle(self, candle_data: Dict) -> bool:
        """
        Añade una nueva vela al buffer con validación.
        
        Args:
            candle_data: Datos de la vela
            
        Returns:
            True si se añadió correctamente, False si falló validación
        """
        try:
            with self.lock:
                # Validar estructura básica
                required_fields = ['open', 'high', 'low', 'close', 'open_time']
                if not all(field in candle_data for field in required_fields):
                    self.logger.warning(f"Vela inválida: faltan campos {required_fields}")
                    return False
                
                # Convertir a Decimal para precisión
                processed_candle = {
                    'open': Decimal(str(candle_data['open'])),
                    'high': Decimal(str(candle_data['high'])),
                    'low': Decimal(str(candle_data['low'])),
                    'close': Decimal(str(candle_data['close'])),
                    'open_time': int(candle_data['open_time']),
                    'volume': Decimal(str(candle_data.get('volume', 0))),
                    'timestamp': datetime.now()
                }
                
                # Validar coherencia de precios
                if not self._validate_candle_consistency(processed_candle):
                    return False
                
                # Detectar gaps de datos
                if self._detect_data_gap(processed_candle):
                    self.data_gaps_detected += 1
                    self.logger.warning(f"Gap de datos detectado en timestamp {processed_candle['open_time']}")
                
                self.candles.append(processed_candle)
                self.total_candles_received += 1
                self.last_update = datetime.now()
                
                # Log periódico
                if self.total_candles_received % 100 == 0:
                    self.logger.debug(f"Buffer: {len(self.candles)} velas, {self.total_candles_received} total recibidas")
                
                return True
                
        except (ValueError, KeyError) as e:
            self.logger.error(f"Error procesando vela: {e}")
            return False
    
    def add_tick(self, tick_data: Dict) -> bool:
        """
        Añade un tick al buffer.
        
        Args:
            tick_data: Datos del tick
            
        Returns:
            True si se añadió correctamente
        """
        try:
            with self.lock:
                processed_tick = {
                    'bid': Decimal(str(tick_data.get('bid', 0))),
                    'ask': Decimal(str(tick_data.get('ask', 0))),
                    'timestamp': tick_data.get('epoch', int(time.time())),
                    'symbol': tick_data.get('symbol'),
                    'received_at': datetime.now()
                }
                
                self.ticks.append(processed_tick)
                self.total_ticks_received += 1
                return True
                
        except (ValueError, KeyError) as e:
            self.logger.error(f"Error procesando tick: {e}")
            return False
    
    def get_recent_candles(self, count: int = 50) -> List[Dict]:
        """
        Obtiene las últimas N velas del buffer.
        
        Args:
            count: Número de velas a obtener
            
        Returns:
            Lista de velas más recientes
        """
        with self.lock:
            if count >= len(self.candles):
                return list(self.candles)
            else:
                return list(self.candles)[-count:]
    
    def get_current_price(self) -> Optional[Decimal]:
        """
        Obtiene el precio actual (último cierre).
        
        Returns:
            Precio actual o None si no hay datos
        """
        with self.lock:
            if self.candles:
                return self.candles[-1]['close']
            return None
    
    def get_price_history(self, field: str = 'close', count: int = 100) -> List[Decimal]:
        """
        Obtiene historial de precios para un campo específico.
        
        Args:
            field: Campo de precio ('open', 'high', 'low', 'close')
            count: Número de valores a obtener
            
        Returns:
            Lista de precios
        """
        with self.lock:
            recent_candles = self.get_recent_candles(count)
            return [candle[field] for candle in recent_candles if field in candle]
    
    def get_ohlc_arrays(self, count: int = 100) -> Dict[str, List[Decimal]]:
        """
        Obtiene arrays OHLC para cálculos de indicadores.
        
        Args:
            count: Número de velas
            
        Returns:
            Diccionario con arrays de open, high, low, close
        """
        with self.lock:
            recent_candles = self.get_recent_candles(count)
            
            return {
                'opens': [c['open'] for c in recent_candles],
                'highs': [c['high'] for c in recent_candles],
                'lows': [c['low'] for c in recent_candles],
                'closes': [c['close'] for c in recent_candles],
                'volumes': [c.get('volume', Decimal('0')) for c in recent_candles],
                'timestamps': [c['open_time'] for c in recent_candles]
            }
    
    def is_data_fresh(self, max_age_seconds: int = 300) -> bool:
        """
        Verifica si los datos están actualizados.
        
        Args:
            max_age_seconds: Edad máxima en segundos
            
        Returns:
            True si los datos están frescos
        """
        if not self.last_update:
            return False
            
        age = (datetime.now() - self.last_update).total_seconds()
        return age <= max_age_seconds
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del buffer.
        
        Returns:
            Diccionario con estadísticas
        """
        with self.lock:
            return {
                'candles_in_buffer': len(self.candles),
                'ticks_in_buffer': len(self.ticks),
                'total_candles_received': self.total_candles_received,
                'total_ticks_received': self.total_ticks_received,
                'data_gaps_detected': self.data_gaps_detected,
                'last_update': self.last_update,
                'symbol': self.symbol,
                'granularity': self.granularity,
                'is_data_fresh': self.is_data_fresh(),
                'current_price': self.get_current_price()
            }
    
    def clear_buffer(self) -> None:
        """Limpia todos los buffers."""
        with self.lock:
            self.candles.clear()
            self.ticks.clear()
            self.last_update = None
            self.logger.info("Buffer limpiado")
    
    def _validate_candle_consistency(self, candle: Dict) -> bool:
        """
        Valida la coherencia interna de una vela.
        
        Args:
            candle: Datos de la vela
            
        Returns:
            True si la vela es consistente
        """
        try:
            o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
            
            # Verificar que high >= max(open, close)
            if h < max(o, c):
                self.logger.warning(f"Vela inválida: high {h} < max(open {o}, close {c})")
                return False
            
            # Verificar que low <= min(open, close)  
            if l > min(o, c):
                self.logger.warning(f"Vela inválida: low {l} > min(open {o}, close {c})")
                return False
            
            # Verificar que todos los precios sean positivos
            if any(price <= 0 for price in [o, h, l, c]):
                self.logger.warning("Vela inválida: precio negativo o cero")
                return False
            
            return True
            
        except (KeyError, TypeError):
            return False
    
    def _detect_data_gap(self, new_candle: Dict) -> bool:
        """
        Detecta gaps en los datos basado en timestamps.
        
        Args:
            new_candle: Nueva vela a verificar
            
        Returns:
            True si se detecta un gap
        """
        if len(self.candles) == 0:
            return False
            
        if not self.granularity:
            return False
            
        last_candle = self.candles[-1]
        expected_next_time = last_candle['open_time'] + self.granularity
        actual_time = new_candle['open_time']
        
        # Permitir un margen de tolerancia (5% del granularidad)
        tolerance = self.granularity * 0.05
        
        return abs(actual_time - expected_next_time) > tolerance


class MarketDataManager:
    """Gestor principal de datos de mercado con múltiples fuentes."""
    
    def __init__(self, symbol: str, granularity: int):
        """
        Inicializa el gestor de datos de mercado.
        
        Args:
            symbol: Símbolo del instrumento
            granularity: Granularidad en segundos
        """
        self.logger = logging.getLogger("MarketDataManager")
        self.symbol = symbol
        self.granularity = granularity
        
        # Buffer principal
        self.buffer = MarketDataBuffer()
        self.buffer.symbol = symbol
        self.buffer.granularity = granularity
        
        # Callbacks para eventos
        self.on_new_candle_callbacks: List[Callable] = []
        self.on_tick_callbacks: List[Callable] = []
        
        # Control de estado
        self.is_active = False
        self.last_error = None
        
        self.logger.info(f"MarketDataManager inicializado para {symbol} ({granularity}s)")
    
    def register_candle_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Registra un callback para nuevas velas.
        
        Args:
            callback: Función a llamar cuando llega una nueva vela
        """
        self.on_new_candle_callbacks.append(callback)
        self.logger.debug(f"Callback de vela registrado: {callback.__name__}")
    
    def register_tick_callback(self, callback: Callable[[Dict], None]) -> None:
        """
        Registra un callback para nuevos ticks.
        
        Args:
            callback: Función a llamar cuando llega un nuevo tick
        """
        self.on_tick_callbacks.append(callback)
        self.logger.debug(f"Callback de tick registrado: {callback.__name__}")
    
    def process_candle_data(self, raw_data: Dict) -> bool:
        """
        Procesa datos de vela recibidos de la fuente externa.
        
        Args:
            raw_data: Datos brutos de la vela
            
        Returns:
            True si se procesó correctamente
        """
        try:
            # Normalizar datos dependiendo de la fuente
            normalized_data = self._normalize_candle_data(raw_data)
            
            if self.buffer.add_candle(normalized_data):
                # Notificar a callbacks
                for callback in self.on_new_candle_callbacks:
                    try:
                        callback(normalized_data)
                    except Exception as e:
                        self.logger.error(f"Error en callback de vela: {e}")
                
                return True
            else:
                self.last_error = "Error añadiendo vela al buffer"
                return False
                
        except Exception as e:
            self.last_error = f"Error procesando vela: {e}"
            self.logger.error(self.last_error)
            return False
    
    def process_tick_data(self, raw_data: Dict) -> bool:
        """
        Procesa datos de tick recibidos.
        
        Args:
            raw_data: Datos brutos del tick
            
        Returns:
            True si se procesó correctamente
        """
        try:
            if self.buffer.add_tick(raw_data):
                # Notificar a callbacks
                for callback in self.on_tick_callbacks:
                    try:
                        callback(raw_data)
                    except Exception as e:
                        self.logger.error(f"Error en callback de tick: {e}")
                
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error procesando tick: {e}")
            return False
    
    def get_latest_candles(self, count: int = 50) -> List[Dict]:
        """
        Obtiene las últimas velas disponibles.
        
        Args:
            count: Número de velas a obtener
            
        Returns:
            Lista de velas
        """
        return self.buffer.get_recent_candles(count)
    
    def get_price_data_for_analysis(self, count: int = 100) -> Dict[str, List[Decimal]]:
        """
        Obtiene datos de precios formateados para análisis técnico.
        
        Args:
            count: Número de velas
            
        Returns:
            Diccionario con arrays OHLC
        """
        return self.buffer.get_ohlc_arrays(count)
    
    def is_data_healthy(self) -> Tuple[bool, str]:
        """
        Verifica el estado de salud de los datos.
        
        Returns:
            Tupla (is_healthy, status_message)
        """
        stats = self.buffer.get_buffer_stats()
        
        # Verificar que tenemos datos
        if stats['candles_in_buffer'] == 0:
            return False, "No hay datos de velas disponibles"
        
        # Verificar frescura de datos
        if not stats['is_data_fresh']:
            return False, "Los datos no están actualizados"
        
        # Verificar muchos gaps de datos
        if stats['data_gaps_detected'] > 10:
            return False, f"Demasiados gaps de datos: {stats['data_gaps_detected']}"
        
        # Verificar precio válido
        if not stats['current_price'] or stats['current_price'] <= 0:
            return False, "Precio actual inválido"
        
        return True, "Datos saludables"
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Genera un reporte completo del estado.
        
        Returns:
            Diccionario con información de estado
        """
        stats = self.buffer.get_buffer_stats()
        is_healthy, health_msg = self.is_data_healthy()
        
        return {
            **stats,
            'is_healthy': is_healthy,
            'health_message': health_msg,
            'last_error': self.last_error,
            'is_active': self.is_active,
            'uptime': (datetime.now() - stats['last_update']).total_seconds() if stats['last_update'] else 0
        }
    
    def _normalize_candle_data(self, raw_data: Dict) -> Dict:
        """
        Normaliza datos de vela desde diferentes fuentes.
        
        Args:
            raw_data: Datos brutos
            
        Returns:
            Datos normalizados
        """
        # Detectar formato de datos (Deriv, Binance, etc.)
        if 'ohlc' in raw_data:  # Formato Deriv
            ohlc = raw_data['ohlc']
            return {
                'open': ohlc.get('open'),
                'high': ohlc.get('high'),
                'low': ohlc.get('low'),
                'close': ohlc.get('close'),
                'open_time': ohlc.get('open_time'),
                'volume': ohlc.get('volume', 0)
            }
        
        # Formato directo (ya normalizado)
        elif all(key in raw_data for key in ['open', 'high', 'low', 'close']):
            return raw_data
        
        # Formato Binance
        elif isinstance(raw_data, list) and len(raw_data) >= 6:
            return {
                'open_time': int(raw_data[0]) // 1000,  # Convertir de ms a s
                'open': float(raw_data[1]),
                'high': float(raw_data[2]),
                'low': float(raw_data[3]),
                'close': float(raw_data[4]),
                'volume': float(raw_data[5])
            }
        
        else:
            raise ValueError(f"Formato de datos no reconocido: {raw_data}")
    
    def cleanup(self) -> None:
        """Limpia recursos del gestor."""
        self.buffer.clear_buffer()
        self.on_new_candle_callbacks.clear()
        self.on_tick_callbacks.clear()
        self.is_active = False
        self.logger.info("MarketDataManager limpiado")