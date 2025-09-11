import time
import logging
from collections import deque
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from .base_strategy import BaseStrategy

class ConfluenceStrategy(BaseStrategy):
    """Analiza el mercado y genera señales de trading optimizado."""
    
    def __init__(self, symbol: str, strategy_params: Dict, tolerancia_zona: Decimal, 
                 ema_fast_period: int, ema_slow_period: int):
        """Inicializa el analizador de estrategia."""
        self.symbol = symbol
        self.strategy_params = strategy_params
        self.tolerancia_zona = tolerancia_zona
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        
        # Inicializar logger PRIMERO
        self.logger = logging.getLogger("ConfluenceStrategy")
        
        # Validar y cargar zonas
        self._load_zones()
        
        # Buffer optimizado para velas
        max_buffer = max(ema_slow_period + 10, 50)
        self.candles = deque(maxlen=max_buffer)
        
        # Cache para EMAs
        self.ema_cache = {'fast': None, 'slow': None, 'last_update': None}
        
        self.last_trade_ts = None
        self.last_signal_ts = None
        self.signal_cooldown = 300  # 5 minutos entre señales

    def _load_zones(self) -> None:
        """Carga y valida las zonas de trading."""
        try:
            supports = self.strategy_params.get('SUPPORTS', [])
            resistances = self.strategy_params.get('RESISTANCES', [])
            monthly_low = self.strategy_params.get('MONTHLY_LOW')
            monthly_high = self.strategy_params.get('MONTHLY_HIGH')
            
            # Agregar niveles mensuales si no están en las listas
            if monthly_low and monthly_low not in supports:
                supports.append(monthly_low)
            if monthly_high and monthly_high not in resistances:
                resistances.append(monthly_high)
            
            self.active_supports = sorted([Decimal(str(s)) for s in supports])
            self.active_resistances = sorted([Decimal(str(r)) for r in resistances], reverse=True)
            
            self.logger.info(f"Zonas cargadas - Soportes: {self.active_supports}")
            self.logger.info(f"Zonas cargadas - Resistencias: {self.active_resistances}")
            
        except Exception as e:
            self.logger.error(f"Error al cargar zonas: {e}")
            raise

    def update_active_zones(self, close_price: Decimal) -> None:
        """Actualiza soportes y resistencias tras rupturas significativas."""
        price_change_threshold = close_price * Decimal('0.001')  # 0.1% como cambio significativo
        
        # Revisar rupturas de soporte
        for support in list(self.active_supports):
            if close_price < (support - price_change_threshold):
                self.logger.warning(f"Ruptura significativa de soporte en {support}")
                self.active_supports.remove(support)
                if support not in self.active_resistances:
                    self.active_resistances.append(support)
                    self.active_resistances.sort(reverse=True)
        
        # Revisar rupturas de resistencia  
        for resistance in list(self.active_resistances):
            if close_price > (resistance + price_change_threshold):
                self.logger.warning(f"Ruptura significativa de resistencia en {resistance}")
                self.active_resistances.remove(resistance)
                if resistance not in self.active_supports:
                    self.active_supports.append(resistance)
                    self.active_supports.sort()

    def should_trade(self, ts: int) -> bool:
        """Evita operar más de una vez por vela y aplica cooldown."""
        current_time = time.time()
        
        # Evitar trade en la misma vela
        if self.last_trade_ts == ts:
            return False
        
        # Cooldown entre señales
        if self.last_signal_ts and (current_time - self.last_signal_ts) < self.signal_cooldown:
            return False
        
        self.last_trade_ts = ts
        self.last_signal_ts = current_time
        return True

    def _validate_candle(self, candle: Dict) -> bool:
        """Valida que la vela tenga datos consistentes."""
        try:
            o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
            return h >= max(o, c) and l <= min(o, c) and all(x > 0 for x in [o, h, l, c])
        except (KeyError, ValueError):
            return False

    # Patrones de velas optimizados con mejor detección
    def is_hammer(self) -> Tuple[bool, str]:
        """Detecta martillo con validación mejorada."""
        if len(self.candles) < 2:
            return False, ""
        
        current, prev = self.candles[-1], self.candles[-2]
        
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        lower_wick = min(current['open'], current['close']) - current['low']
        upper_wick = current['high'] - max(current['open'], current['close'])
        
        if total_range == 0:
            return False, ""
        
        # Condiciones del martillo
        is_bullish = current['close'] > current['open']
        long_lower_wick = lower_wick >= (body_size * Decimal('2'))
        small_upper_wick = upper_wick <= (body_size * Decimal('0.3'))
        significant_wick = lower_wick >= (total_range * Decimal('0.6'))
        
        # Contexto bajista previo
        prev_bearish = prev['close'] < prev['open']
        
        if is_bullish and long_lower_wick and small_upper_wick and significant_wick and prev_bearish:
            confidence = min(lower_wick / body_size, Decimal('5'))
            return True, f"Martillo (confianza: {confidence:.1f})"
        
        return False, ""

    def is_bullish_engulfing(self) -> Tuple[bool, str]:
        """Detecta envolvente alcista con validación de contexto."""
        if len(self.candles) < 3:
            return False, ""
        
        prev_prev, prev, current = self.candles[-3], self.candles[-2], self.candles[-1]
        
        # Condiciones básicas
        prev_bearish = prev['close'] < prev['open']
        current_bullish = current['close'] > current['open']
        engulfs = current['open'] < prev['close'] and current['close'] > prev['open']
        
        # Validar contexto de tendencia bajista
        downtrend_context = (prev_prev['close'] > prev['close'] and 
                           prev['close'] < prev['open'])
        
        # Validar volumen relativo (usando rango como proxy)
        current_range = current['high'] - current['low']
        prev_range = prev['high'] - prev['low']
        good_volume = current_range >= (prev_range * Decimal('1.2'))
        
        if prev_bearish and current_bullish and engulfs and downtrend_context and good_volume:
            body_ratio = abs(current['close'] - current['open']) / abs(prev['close'] - prev['open'])
            return True, f"Envolvente Alcista (ratio: {body_ratio:.1f})"
        
        return False, ""

    def is_bearish_engulfing(self) -> Tuple[bool, str]:
        """Detecta envolvente bajista con validación de contexto."""
        if len(self.candles) < 3:
            return False, ""
        
        prev_prev, prev, current = self.candles[-3], self.candles[-2], self.candles[-1]
        
        # Condiciones básicas
        prev_bullish = prev['close'] > prev['open']
        current_bearish = current['close'] < current['open']
        engulfs = current['open'] > prev['close'] and current['close'] < prev['open']
        
        # Validar contexto de tendencia alcista
        uptrend_context = (prev_prev['close'] < prev['close'] and 
                          prev['close'] > prev['open'])
        
        # Validar volumen relativo
        current_range = current['high'] - current['low']
        prev_range = prev['high'] - prev['low']
        good_volume = current_range >= (prev_range * Decimal('1.2'))
        
        if prev_bullish and current_bearish and engulfs and uptrend_context and good_volume:
            body_ratio = abs(current['close'] - current['open']) / abs(prev['close'] - prev['open'])
            return True, f"Envolvente Bajista (ratio: {body_ratio:.1f})"
        
        return False, ""

    def is_inside_bar(self) -> Tuple[bool, str]:
        """Detecta barra interna con análisis de contexto."""
        if len(self.candles) < 3:
            return False, ""
        
        mother_bar, inside_bar = self.candles[-2], self.candles[-1]
        
        # Condiciones de barra interna
        is_inside = (inside_bar['high'] < mother_bar['high'] and 
                    inside_bar['low'] > mother_bar['low'])
        
        if not is_inside:
            return False, ""
        
        # Evaluar la fuerza de la barra madre
        mother_range = mother_bar['high'] - mother_bar['low']
        inside_range = inside_bar['high'] - inside_bar['low']
        
        compression_ratio = inside_range / mother_range if mother_range > 0 else 0
        
        # Barra interna significativa (compresión >= 50%)
        if compression_ratio <= Decimal('0.5'):
            return True, f"Inside Bar (compresión: {compression_ratio:.1%})"
        
        return False, ""

    def calculate_ema_optimized(self, period: int) -> Optional[Decimal]:
        """Calcula EMA usando cache para optimización."""
        if len(self.candles) < period:
            return None
        
        cache_key = f"{period}_{len(self.candles)}"
        
        # Verificar cache
        if (self.ema_cache.get('last_update') == cache_key and 
            self.ema_cache.get(f'ema_{period}')):
            return self.ema_cache[f'ema_{period}']
        
        prices = [c['close'] for c in self.candles]
        
        # Calcular SMA inicial
        sma = sum(prices[:period]) / period
        ema = sma
        
        # Calcular EMA
        multiplier = Decimal('2') / (period + 1)
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        # Actualizar cache
        self.ema_cache[f'ema_{period}'] = ema
        self.ema_cache['last_update'] = cache_key
        
        return ema

    def get_trend_context(self) -> str:
        """Determina el contexto de tendencia actual."""
        if len(self.candles) < 10:
            return "insuficiente"
        
        recent_closes = [c['close'] for c in list(self.candles)[-10:]]
        
        # Tendencia simple basada en pendiente
        first_half_avg = sum(recent_closes[:5]) / 5
        second_half_avg = sum(recent_closes[5:]) / 5
        
        change_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        if change_pct > Decimal('0.05'):  # 0.05%
            return "alcista"
        elif change_pct < Decimal('-0.05'):  # -0.05%
            return "bajista"
        else:
            return "lateral"

    def analyze(self, new_candle: Dict) -> Optional[Tuple[str, str, Dict]]:
        """Analiza el mercado y genera señal con información detallada."""
        # Validar vela
        if not self._validate_candle(new_candle):
            self.logger.warning("Vela inválida recibida")
            return None
        
        # Convertir a Decimal
        try:
            processed_candle = {
                'open': Decimal(str(new_candle['open'])),
                'high': Decimal(str(new_candle['high'])),
                'low': Decimal(str(new_candle['low'])),
                'close': Decimal(str(new_candle['close']))
            }
        except (ValueError, KeyError) as e:
            self.logger.error(f"Error al procesar vela: {e}")
            return None
        
        self.candles.append(processed_candle)
        
        # Verificar datos suficientes
        if len(self.candles) < max(self.ema_slow_period, 10):
            return None
        
        close_price = processed_candle['close']
        self.update_active_zones(close_price)
        
        # Calcular EMAs
        ema_fast = self.calculate_ema_optimized(self.ema_fast_period)
        ema_slow = self.calculate_ema_optimized(self.ema_slow_period)
        
        if ema_fast is None or ema_slow is None:
            return None
        
        # Contexto de mercado
        trend_context = self.get_trend_context()
        
        # Información de análisis
        analysis_info = {
            'close_price': close_price,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'trend_context': trend_context,
            'active_supports': list(self.active_supports),
            'active_resistances': list(self.active_resistances)
        }
        
        # Buscar señales en soportes (CALL)
        for zona in self.active_supports:
            price_diff = abs(close_price - zona)
            zona_tolerance = zona * self.tolerancia_zona
            
            if price_diff <= zona_tolerance:
                # Filtro de tendencia para CALL
                ema_bullish = ema_fast >= ema_slow
                trend_favorable = trend_context in ['alcista', 'lateral']
                
                if ema_bullish and trend_favorable:
                    # Buscar patrones alcistas
                    patterns_found = []
                    
                    hammer_result = self.is_hammer()
                    if hammer_result[0]:
                        patterns_found.append(hammer_result[1])
                    
                    engulfing_result = self.is_bullish_engulfing()
                    if engulfing_result[0]:
                        patterns_found.append(engulfing_result[1])
                    
                    inside_bar_result = self.is_inside_bar()
                    if inside_bar_result[0]:
                        patterns_found.append(inside_bar_result[1])
                    
                    if patterns_found:
                        signal_strength = len(patterns_found)
                        reason = (f"CALL en soporte {zona} | Patrones: {', '.join(patterns_found)} | "
                                f"EMA: {ema_fast:.5f}>{ema_slow:.5f} | Tendencia: {trend_context}")
                        
                        analysis_info.update({
                            'signal_zone': zona,
                            'zone_type': 'soporte',
                            'patterns_found': patterns_found,
                            'signal_strength': signal_strength
                        })
                        
                        self.logger.info(f"🟢 {reason}")
                        return "CALL", reason, analysis_info
        
        # Buscar señales en resistencias (PUT)
        for zona in self.active_resistances:
            price_diff = abs(close_price - zona)
            zona_tolerance = zona * self.tolerancia_zona
            
            if price_diff <= zona_tolerance:
                # Filtro de tendencia para PUT
                ema_bearish = ema_fast <= ema_slow
                trend_favorable = trend_context in ['bajista', 'lateral']
                
                if ema_bearish and trend_favorable:
                    # Buscar patrones bajistas
                    patterns_found = []
                    
                    bearish_engulfing_result = self.is_bearish_engulfing()
                    if bearish_engulfing_result[0]:
                        patterns_found.append(bearish_engulfing_result[1])
                    
                    inside_bar_result = self.is_inside_bar()
                    if inside_bar_result[0]:
                        patterns_found.append(inside_bar_result[1])
                    
                    if patterns_found:
                        signal_strength = len(patterns_found)
                        reason = (f"PUT en resistencia {zona} | Patrones: {', '.join(patterns_found)} | "
                                f"EMA: {ema_fast:.5f}<{ema_slow:.5f} | Tendencia: {trend_context}")
                        
                        analysis_info.update({
                            'signal_zone': zona,
                            'zone_type': 'resistencia',
                            'patterns_found': patterns_found,
                            'signal_strength': signal_strength
                        })
                        
                        self.logger.info(f"🔴 {reason}")
                        return "PUT", reason, analysis_info
        
        return None