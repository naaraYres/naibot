"""
Indicadores técnicos para análisis de trading.
Incluye EMAs, RSI, MACD y otros indicadores comunes.
"""

from decimal import Decimal, getcontext
from typing import List, Optional, Tuple, Dict
from collections import deque
import logging

# Configurar precisión decimal
getcontext().prec = 28

class TechnicalIndicators:
    """Calculadora de indicadores técnicos con optimización de caché."""
    
    def __init__(self):
        """Inicializa el calculador de indicadores técnicos."""
        self.logger = logging.getLogger("TechnicalIndicators")
        self._ema_cache: Dict[str, Dict] = {}
        
    def calculate_ema(self, prices: List[Decimal], period: int, 
                     cache_key: Optional[str] = None) -> Optional[Decimal]:
        """
        Calcula EMA (Exponential Moving Average) con caché opcional.
        
        Args:
            prices: Lista de precios
            period: Período para el cálculo
            cache_key: Clave para caché (opcional)
            
        Returns:
            Valor de EMA o None si no hay suficientes datos
        """
        if len(prices) < period:
            return None
            
        # Verificar cache si se proporciona clave
        if cache_key and cache_key in self._ema_cache:
            cache_data = self._ema_cache[cache_key]
            if (cache_data.get('last_period') == period and 
                cache_data.get('last_length') == len(prices) - 1):
                
                # Calcular solo para el último precio
                last_ema = cache_data['last_ema']
                multiplier = Decimal('2') / (period + 1)
                new_ema = (prices[-1] - last_ema) * multiplier + last_ema
                
                # Actualizar cache
                self._ema_cache[cache_key] = {
                    'last_ema': new_ema,
                    'last_period': period,
                    'last_length': len(prices)
                }
                
                return new_ema
        
        # Calcular EMA completa
        sma = sum(prices[:period]) / period
        ema = sma
        
        multiplier = Decimal('2') / (period + 1)
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
            
        # Guardar en cache si se proporciona clave
        if cache_key:
            self._ema_cache[cache_key] = {
                'last_ema': ema,
                'last_period': period,
                'last_length': len(prices)
            }
            
        return ema
    
    def calculate_sma(self, prices: List[Decimal], period: int) -> Optional[Decimal]:
        """
        Calcula SMA (Simple Moving Average).
        
        Args:
            prices: Lista de precios
            period: Período para el cálculo
            
        Returns:
            Valor de SMA o None si no hay suficientes datos
        """
        if len(prices) < period:
            return None
            
        return sum(prices[-period:]) / period
    
    def calculate_rsi(self, prices: List[Decimal], period: int = 14) -> Optional[Decimal]:
        """
        Calcula RSI (Relative Strength Index).
        
        Args:
            prices: Lista de precios de cierre
            period: Período para el cálculo (default: 14)
            
        Returns:
            Valor de RSI (0-100) o None si no hay suficientes datos
        """
        if len(prices) < period + 1:
            return None
            
        # Calcular cambios de precio
        price_changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            price_changes.append(change)
        
        if len(price_changes) < period:
            return None
            
        # Separar ganancias y pérdidas
        gains = [max(change, Decimal('0')) for change in price_changes[-period:]]
        losses = [abs(min(change, Decimal('0'))) for change in price_changes[-period:]]
        
        # Calcular promedios
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        # Evitar división por cero
        if avg_loss == 0:
            return Decimal('100')
            
        # Calcular RSI
        rs = avg_gain / avg_loss
        rsi = Decimal('100') - (Decimal('100') / (Decimal('1') + rs))
        
        return rsi
    
    def calculate_macd(self, prices: List[Decimal], 
                      fast_period: int = 12, slow_period: int = 26, 
                      signal_period: int = 9) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
        """
        Calcula MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Lista de precios de cierre
            fast_period: Período EMA rápida (default: 12)
            slow_period: Período EMA lenta (default: 26)
            signal_period: Período señal MACD (default: 9)
            
        Returns:
            Tupla (MACD, Signal, Histogram) o None si no hay suficientes datos
        """
        if len(prices) < slow_period:
            return None
            
        # Calcular EMAs
        ema_fast = self.calculate_ema(prices, fast_period, f"macd_fast_{fast_period}")
        ema_slow = self.calculate_ema(prices, slow_period, f"macd_slow_{slow_period}")
        
        if ema_fast is None or ema_slow is None:
            return None
            
        # Calcular línea MACD
        macd_line = ema_fast - ema_slow
        
        # Para calcular la señal, necesitamos un historial de valores MACD
        # Simplificado: usar diferencia directa
        macd_values = []
        for i in range(slow_period, len(prices) + 1):
            subset = prices[:i]
            ema_f = self.calculate_ema(subset, fast_period)
            ema_s = self.calculate_ema(subset, slow_period)
            if ema_f is not None and ema_s is not None:
                macd_values.append(ema_f - ema_s)
        
        if len(macd_values) < signal_period:
            signal_line = macd_line  # Fallback
        else:
            signal_line = self.calculate_ema(macd_values, signal_period, f"macd_signal_{signal_period}")
            if signal_line is None:
                signal_line = macd_line
                
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: List[Decimal], period: int = 20, 
                                 std_dev: Decimal = Decimal('2')) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
        """
        Calcula Bandas de Bollinger.
        
        Args:
            prices: Lista de precios
            period: Período para SMA y desviación estándar
            std_dev: Multiplicador de desviación estándar
            
        Returns:
            Tupla (Upper Band, Middle Band/SMA, Lower Band) o None
        """
        if len(prices) < period:
            return None
            
        # Calcular SMA
        sma = self.calculate_sma(prices, period)
        if sma is None:
            return None
            
        # Calcular desviación estándar
        recent_prices = prices[-period:]
        variance = sum((price - sma) ** 2 for price in recent_prices) / period
        std = variance.sqrt()
        
        # Calcular bandas
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_stochastic(self, highs: List[Decimal], lows: List[Decimal], 
                           closes: List[Decimal], k_period: int = 14, 
                           d_period: int = 3) -> Optional[Tuple[Decimal, Decimal]]:
        """
        Calcula Oscilador Estocástico.
        
        Args:
            highs: Lista de precios máximos
            lows: Lista de precios mínimos  
            closes: Lista de precios de cierre
            k_period: Período para %K
            d_period: Período para %D (SMA de %K)
            
        Returns:
            Tupla (%K, %D) o None si no hay suficientes datos
        """
        if len(closes) < k_period or len(highs) < k_period or len(lows) < k_period:
            return None
            
        # Calcular %K
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = Decimal('50')  # Valor neutral
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * Decimal('100')
        
        # Para %D necesitamos varios valores de %K
        k_values = []
        for i in range(k_period, len(closes) + 1):
            subset_highs = highs[i-k_period:i]
            subset_lows = lows[i-k_period:i]
            subset_close = closes[i-1]
            
            h_high = max(subset_highs)
            l_low = min(subset_lows)
            
            if h_high == l_low:
                k_val = Decimal('50')
            else:
                k_val = ((subset_close - l_low) / (h_high - l_low)) * Decimal('100')
            k_values.append(k_val)
        
        # Calcular %D como SMA de %K
        if len(k_values) < d_period:
            d_percent = k_percent
        else:
            d_percent = sum(k_values[-d_period:]) / d_period
            
        return k_percent, d_percent
    
    def is_oversold(self, rsi_value: Decimal, threshold: Decimal = Decimal('30')) -> bool:
        """Determina si el RSI indica sobreventa."""
        return rsi_value < threshold
    
    def is_overbought(self, rsi_value: Decimal, threshold: Decimal = Decimal('70')) -> bool:
        """Determina si el RSI indica sobrecompra."""
        return rsi_value > threshold
    
    def get_ema_trend(self, ema_fast: Decimal, ema_slow: Decimal) -> str:
        """
        Determina la tendencia basada en EMAs.
        
        Returns:
            'bullish', 'bearish', o 'neutral'
        """
        if ema_fast > ema_slow:
            return 'bullish'
        elif ema_fast < ema_slow:
            return 'bearish'
        else:
            return 'neutral'
    
    def get_macd_signal(self, macd_line: Decimal, signal_line: Decimal, 
                       histogram: Decimal) -> str:
        """
        Interpreta la señal MACD.
        
        Returns:
            'bullish', 'bearish', o 'neutral'
        """
        if macd_line > signal_line and histogram > 0:
            return 'bullish'
        elif macd_line < signal_line and histogram < 0:
            return 'bearish'
        else:
            return 'neutral'
    
    def clear_cache(self) -> None:
        """Limpia el caché de EMAs."""
        self._ema_cache.clear()
        self.logger.debug("Cache de indicadores limpiado")


class CandlestickPatterns:
    """Detector de patrones de velas japonesas."""
    
    def __init__(self):
        """Inicializa el detector de patrones."""
        self.logger = logging.getLogger("CandlestickPatterns")
    
    @staticmethod
    def _validate_candle(candle: Dict) -> bool:
        """Valida que la vela tenga datos consistentes."""
        try:
            o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
            return h >= max(o, c) and l <= min(o, c) and all(x > 0 for x in [o, h, l, c])
        except (KeyError, ValueError):
            return False
    
    @staticmethod
    def _get_candle_metrics(candle: Dict) -> Dict[str, Decimal]:
        """Obtiene métricas básicas de una vela."""
        o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
        
        body_size = abs(c - o)
        total_range = h - l
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        return {
            'body_size': body_size,
            'total_range': total_range,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'is_bullish': c > o,
            'is_bearish': c < o,
            'is_doji': body_size <= (total_range * Decimal('0.1')) if total_range > 0 else True
        }
    
    def is_hammer(self, candles: List[Dict]) -> Tuple[bool, str]:
        """Detecta patrón de martillo."""
        if len(candles) < 2:
            return False, ""
            
        current, previous = candles[-1], candles[-2]
        
        if not (self._validate_candle(current) and self._validate_candle(previous)):
            return False, ""
            
        current_metrics = self._get_candle_metrics(current)
        previous_metrics = self._get_candle_metrics(previous)
        
        # Condiciones del martillo
        conditions = [
            current_metrics['is_bullish'],  # Vela actual alcista
            previous_metrics['is_bearish'],  # Vela previa bajista
            current_metrics['lower_wick'] >= (current_metrics['body_size'] * 2),  # Mecha inferior larga
            current_metrics['upper_wick'] <= (current_metrics['body_size'] * 0.3),  # Mecha superior corta
            current_metrics['total_range'] > 0  # Evitar división por cero
        ]
        
        if all(conditions):
            confidence = min(current_metrics['lower_wick'] / current_metrics['body_size'], Decimal('5'))
            return True, f"Hammer (confidence: {confidence:.1f})"
            
        return False, ""
    
    def is_bullish_engulfing(self, candles: List[Dict]) -> Tuple[bool, str]:
        """Detecta patrón envolvente alcista."""
        if len(candles) < 2:
            return False, ""
            
        previous, current = candles[-2], candles[-1]
        
        if not (self._validate_candle(current) and self._validate_candle(previous)):
            return False, ""
            
        current_metrics = self._get_candle_metrics(current)
        previous_metrics = self._get_candle_metrics(previous)
        
        # Condiciones del engulfing alcista
        conditions = [
            previous_metrics['is_bearish'],  # Vela previa bajista
            current_metrics['is_bullish'],   # Vela actual alcista
            current['open'] < previous['close'],  # Apertura actual < cierre previo
            current['close'] > previous['open']   # Cierre actual > apertura previa
        ]
        
        if all(conditions):
            body_ratio = current_metrics['body_size'] / previous_metrics['body_size'] if previous_metrics['body_size'] > 0 else 1
            return True, f"Bullish Engulfing (ratio: {body_ratio:.1f})"
            
        return False, ""
    
    def is_bearish_engulfing(self, candles: List[Dict]) -> Tuple[bool, str]:
        """Detecta patrón envolvente bajista."""
        if len(candles) < 2:
            return False, ""
            
        previous, current = candles[-2], candles[-1]
        
        if not (self._validate_candle(current) and self._validate_candle(previous)):
            return False, ""
            
        current_metrics = self._get_candle_metrics(current)
        previous_metrics = self._get_candle_metrics(previous)
        
        # Condiciones del engulfing bajista
        conditions = [
            previous_metrics['is_bullish'],  # Vela previa alcista
            current_metrics['is_bearish'],   # Vela actual bajista
            current['open'] > previous['close'],  # Apertura actual > cierre previo
            current['close'] < previous['open']   # Cierre actual < apertura previa
        ]
        
        if all(conditions):
            body_ratio = current_metrics['body_size'] / previous_metrics['body_size'] if previous_metrics['body_size'] > 0 else 1
            return True, f"Bearish Engulfing (ratio: {body_ratio:.1f})"
            
        return False, ""
    
    def is_inside_bar(self, candles: List[Dict]) -> Tuple[bool, str]:
        """Detecta patrón de barra interna."""
        if len(candles) < 2:
            return False, ""
            
        mother_bar, inside_bar = candles[-2], candles[-1]
        
        if not (self._validate_candle(inside_bar) and self._validate_candle(mother_bar)):
            return False, ""
            
        # Condiciones de inside bar
        conditions = [
            inside_bar['high'] < mother_bar['high'],  # Máximo menor
            inside_bar['low'] > mother_bar['low']     # Mínimo mayor
        ]
        
        if all(conditions):
            mother_range = mother_bar['high'] - mother_bar['low']
            inside_range = inside_bar['high'] - inside_bar['low']
            
            compression = inside_range / mother_range if mother_range > 0 else 0
            return True, f"Inside Bar (compression: {compression:.1%})"
            
        return False, ""
    
    def is_doji(self, candle: Dict) -> Tuple[bool, str]:
        """Detecta patrón Doji."""
        if not self._validate_candle(candle):
            return False, ""
            
        metrics = self._get_candle_metrics(candle)
        
        if metrics['is_doji']:
            if metrics['total_range'] > 0:
                body_ratio = metrics['body_size'] / metrics['total_range']
                return True, f"Doji (body ratio: {body_ratio:.1%})"
            else:
                return True, "Doji (perfect)"
                
        return False, ""