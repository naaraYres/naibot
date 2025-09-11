"""
Indicadores técnicos personalizados para estrategias específicas.
"""

from decimal import Decimal, getcontext
from typing import List, Optional, Dict, Tuple
import logging

# Configurar precisión decimal
getcontext().prec = 28

class CustomIndicators:
    """Indicadores personalizados para trading."""
    
    def __init__(self):
        """Inicializa los indicadores personalizados."""
        self.logger = logging.getLogger("CustomIndicators")
    
    def confluence_strength(self, close_price: Decimal, zones: List[Decimal], 
                          tolerance: Decimal) -> Tuple[int, List[Decimal]]:
        """
        Calcula la fuerza de confluencia cerca de zonas clave.
        
        Args:
            close_price: Precio de cierre actual
            zones: Lista de zonas de soporte/resistencia
            tolerance: Tolerancia para considerar proximidad a zona
            
        Returns:
            Tupla (fuerza_confluencia, zonas_activas)
        """
        active_zones = []
        
        for zone in zones:
            zone_tolerance = zone * tolerance
            price_diff = abs(close_price - zone)
            
            if price_diff <= zone_tolerance:
                active_zones.append(zone)
        
        confluence_strength = len(active_zones)
        
        if confluence_strength > 0:
            self.logger.debug(f"Confluencia detectada: {confluence_strength} zonas activas")
            
        return confluence_strength, active_zones
    
    def momentum_oscillator(self, prices: List[Decimal], period: int = 14) -> Optional[Decimal]:
        """
        Calcula un oscilador de momentum personalizado.
        
        Args:
            prices: Lista de precios
            period: Período de cálculo
            
        Returns:
            Valor del oscilador (-100 a +100) o None
        """
        if len(prices) < period + 1:
            return None
            
        current_price = prices[-1]
        past_price = prices[-(period + 1)]
        
        if past_price == 0:
            return None
            
        # Calcular cambio porcentual
        change_pct = ((current_price - past_price) / past_price) * 100
        
        # Normalizar a rango -100 a +100
        normalized = max(min(change_pct, Decimal('100')), Decimal('-100'))
        
        return normalized
    
    def volatility_index(self, highs: List[Decimal], lows: List[Decimal], 
                        period: int = 14) -> Optional[Decimal]:
        """
        Calcula un índice de volatilidad personalizado.
        
        Args:
            highs: Lista de precios máximos
            lows: Lista de precios mínimos
            period: Período de cálculo
            
        Returns:
            Índice de volatilidad o None
        """
        if len(highs) < period or len(lows) < period:
            return None
            
        # Calcular rangos verdaderos
        true_ranges = []
        for i in range(-period, 0):
            daily_range = highs[i] - lows[i]
            true_ranges.append(daily_range)
        
        if not true_ranges:
            return None
            
        # Promedio de rangos verdaderos
        avg_true_range = sum(true_ranges) / len(true_ranges)
        
        # Normalizar basado en precio actual
        current_price = (highs[-1] + lows[-1]) / 2
        if current_price > 0:
            volatility_index = (avg_true_range / current_price) * 100
            return volatility_index
            
        return None
    
    def trend_strength_indicator(self, prices: List[Decimal], 
                               short_period: int = 8, 
                               long_period: int = 21) -> Optional[Decimal]:
        """
        Calcula la fuerza de la tendencia basada en EMAs.
        
        Args:
            prices: Lista de precios
            short_period: Período EMA corta
            long_period: Período EMA larga
            
        Returns:
            Fuerza de tendencia (-100 a +100) o None
        """
        if len(prices) < long_period:
            return None
            
        # Calcular EMAs simples para este indicador
        def simple_ema(data: List[Decimal], period: int) -> Optional[Decimal]:
            if len(data) < period:
                return None
            sma = sum(data[:period]) / period
            ema = sma
            multiplier = Decimal('2') / (period + 1)
            for price in data[period:]:
                ema = (price - ema) * multiplier + ema
            return ema
        
        ema_short = simple_ema(prices, short_period)
        ema_long = simple_ema(prices, long_period)
        
        if ema_short is None or ema_long is None:
            return None
            
        # Calcular diferencia porcentual
        if ema_long > 0:
            diff_pct = ((ema_short - ema_long) / ema_long) * 100
            # Normalizar a rango -100 a +100
            trend_strength = max(min(diff_pct * 10, Decimal('100')), Decimal('-100'))
            return trend_strength
            
        return None
    
    def support_resistance_strength(self, price: Decimal, zone: Decimal, 
                                  historical_touches: int = 0) -> Decimal:
        """
        Calcula la fuerza de una zona de soporte/resistencia.
        
        Args:
            price: Precio actual
            zone: Nivel de zona
            historical_touches: Número de toques históricos
            
        Returns:
            Puntuación de fuerza (0-100)
        """
        # Fuerza base por proximidad
        if zone > 0:
            proximity = abs(price - zone) / zone
            proximity_score = max(0, Decimal('100') * (1 - proximity * 10))
        else:
            proximity_score = Decimal('0')
        
        # Bonus por toques históricos
        historical_bonus = min(historical_touches * 10, 30)
        
        # Puntuación total
        total_strength = min(proximity_score + historical_bonus, Decimal('100'))
        
        return total_strength
    
    def market_phase_detector(self, prices: List[Decimal], 
                            volume_proxy: Optional[List[Decimal]] = None,
                            period: int = 20) -> str:
        """
        Detecta la fase del mercado (trending, ranging, breakout).
        
        Args:
            prices: Lista de precios
            volume_proxy: Proxy de volumen (opcional, puede ser rango de velas)
            period: Período de análisis
            
        Returns:
            Fase detectada: 'trending', 'ranging', 'breakout', 'unknown'
        """
        if len(prices) < period:
            return 'unknown'
            
        recent_prices = prices[-period:]
        
        # Calcular volatilidad
        price_changes = []
        for i in range(1, len(recent_prices)):
            change = abs(recent_prices[i] - recent_prices[i-1])
            price_changes.append(change)
        
        if not price_changes:
            return 'unknown'
            
        avg_change = sum(price_changes) / len(price_changes)
        recent_change = price_changes[-5:] if len(price_changes) >= 5 else price_changes
        recent_avg_change = sum(recent_change) / len(recent_change)
        
        # Calcular tendencia
        first_half = sum(recent_prices[:period//2]) / (period//2)
        second_half = sum(recent_prices[period//2:]) / (len(recent_prices) - period//2)
        
        trend_strength = abs(second_half - first_half) / first_half if first_half > 0 else 0
        
        # Clasificar fase
        if trend_strength > Decimal('0.002'):  # 0.2% de movimiento direccional
            if recent_avg_change > avg_change * Decimal('1.5'):
                return 'breakout'
            else:
                return 'trending'
        else:
            return 'ranging'
    
    def confluence_score(self, close_price: Decimal, 
                        supports: List[Decimal], 
                        resistances: List[Decimal],
                        tolerance: Decimal,
                        ema_signal: str = 'neutral',
                        rsi_signal: str = 'neutral') -> Dict[str, any]:
        """
        Calcula un score de confluencia total.
        
        Args:
            close_price: Precio de cierre actual
            supports: Lista de soportes
            resistances: Lista de resistencias  
            tolerance: Tolerancia para zonas
            ema_signal: Señal de EMAs ('bullish', 'bearish', 'neutral')
            rsi_signal: Señal de RSI ('oversold', 'overbought', 'neutral')
            
        Returns:
            Diccionario con score y detalles
        """
        score = 0
        details = {}
        
        # Confluencia de soportes
        support_confluence, active_supports = self.confluence_strength(
            close_price, supports, tolerance
        )
        
        # Confluencia de resistencias  
        resistance_confluence, active_resistances = self.confluence_strength(
            close_price, resistances, tolerance
        )
        
        # Puntuación base por zonas
        if support_confluence > 0:
            score += support_confluence * 20  # Máximo 20 pts por zona de soporte
            details['signal_bias'] = 'bullish'
            details['active_zones'] = active_supports
            details['zone_type'] = 'support'
            
        if resistance_confluence > 0:
            score += resistance_confluence * 20  # Máximo 20 pts por zona de resistencia
            details['signal_bias'] = 'bearish'
            details['active_zones'] = active_resistances
            details['zone_type'] = 'resistance'
        
        # Bonus por confluencia de múltiples indicadores
        if ema_signal == 'bullish' and support_confluence > 0:
            score += 15
            details['ema_confluence'] = True
        elif ema_signal == 'bearish' and resistance_confluence > 0:
            score += 15
            details['ema_confluence'] = True
        
        if rsi_signal == 'oversold' and support_confluence > 0:
            score += 10
            details['rsi_confluence'] = True
        elif rsi_signal == 'overbought' and resistance_confluence > 0:
            score += 10
            details['rsi_confluence'] = True
        
        # Penalización por señales contradictorias
        if (ema_signal == 'bullish' and resistance_confluence > 0) or \
           (ema_signal == 'bearish' and support_confluence > 0):
            score -= 20
            details['conflicting_signals'] = True
        
        details['total_score'] = score
        details['support_confluence'] = support_confluence
        details['resistance_confluence'] = resistance_confluence
        
        return details
    
    def adaptive_threshold(self, base_threshold: Decimal, 
                         volatility_factor: Decimal,
                         market_phase: str = 'unknown') -> Decimal:
        """
        Calcula un umbral adaptativo basado en condiciones de mercado.
        
        Args:
            base_threshold: Umbral base
            volatility_factor: Factor de volatilidad actual
            market_phase: Fase del mercado
            
        Returns:
            Umbral ajustado
        """
        adjusted_threshold = base_threshold
        
        # Ajustar por volatilidad
        if volatility_factor > Decimal('2'):  # Alta volatilidad
            adjusted_threshold *= Decimal('1.5')
        elif volatility_factor < Decimal('0.5'):  # Baja volatilidad
            adjusted_threshold *= Decimal('0.8')
        
        # Ajustar por fase de mercado
        if market_phase == 'ranging':
            adjusted_threshold *= Decimal('0.7')  # Ser más sensible en ranging
        elif market_phase == 'breakout':
            adjusted_threshold *= Decimal('1.3')  # Ser más conservador en breakouts
        
        return adjusted_threshold


class ZoneAnalyzer:
    """Analizador especializado para zonas de soporte y resistencia."""
    
    def __init__(self):
        """Inicializa el analizador de zonas."""
        self.logger = logging.getLogger("ZoneAnalyzer")
        self.zone_history = {}  # Historial de toques por zona
    
    def update_zone_history(self, price: Decimal, zones: List[Decimal], 
                          tolerance: Decimal) -> None:
        """
        Actualiza el historial de toques de zonas.
        
        Args:
            price: Precio actual
            zones: Lista de zonas
            tolerance: Tolerancia para considerar un toque
        """
        for zone in zones:
            zone_key = str(zone)
            zone_tolerance = zone * tolerance
            
            if abs(price - zone) <= zone_tolerance:
                if zone_key not in self.zone_history:
                    self.zone_history[zone_key] = {
                        'touches': 0,
                        'last_touch': None,
                        'strength': 0
                    }
                
                self.zone_history[zone_key]['touches'] += 1
                self.zone_history[zone_key]['last_touch'] = price
                # Incrementar fuerza basada en número de toques
                self.zone_history[zone_key]['strength'] = min(
                    self.zone_history[zone_key]['touches'] * 10, 100
                )
    
    def get_zone_strength(self, zone: Decimal) -> int:
        """
        Obtiene la fuerza histórica de una zona.
        
        Args:
            zone: Zona a consultar
            
        Returns:
            Fuerza de la zona (0-100)
        """
        zone_key = str(zone)
        return self.zone_history.get(zone_key, {}).get('strength', 0)
    
    def identify_key_levels(self, highs: List[Decimal], lows: List[Decimal],
                          min_touches: int = 2, 
                          tolerance_pct: Decimal = Decimal('0.001')) -> Dict[str, List[Decimal]]:
        """
        Identifica niveles clave de soporte y resistencia automáticamente.
        
        Args:
            highs: Lista de máximos
            lows: Lista de mínimos
            min_touches: Mínimo número de toques para considerar un nivel
            tolerance_pct: Tolerancia porcentual para agrupar niveles
            
        Returns:
            Diccionario con soportes y resistencias identificados
        """
        if len(highs) < 10 or len(lows) < 10:
            return {'supports': [], 'resistances': []}
        
        # Identificar potenciales resistencias (máximos locales)
        potential_resistances = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                potential_resistances.append(highs[i])
        
        # Identificar potenciales soportes (mínimos locales)
        potential_supports = []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                potential_supports.append(lows[i])
        
        # Agrupar niveles similares
        def group_levels(levels: List[Decimal], tolerance: Decimal) -> List[Decimal]:
            if not levels:
                return []
                
            grouped = []
            sorted_levels = sorted(levels)
            
            current_group = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                # Si el nivel está dentro de la tolerancia del grupo actual
                if abs(level - current_group[0]) / current_group[0] <= tolerance:
                    current_group.append(level)
                else:
                    # Si el grupo tiene suficientes toques, agregarlo
                    if len(current_group) >= min_touches:
                        group_avg = sum(current_group) / len(current_group)
                        grouped.append(group_avg)
                    
                    # Comenzar nuevo grupo
                    current_group = [level]
            
            # Procesar último grupo
            if len(current_group) >= min_touches:
                group_avg = sum(current_group) / len(current_group)
                grouped.append(group_avg)
            
            return grouped
        
        key_supports = group_levels(potential_supports, tolerance_pct)
        key_resistances = group_levels(potential_resistances, tolerance_pct)
        
        self.logger.info(f"Identificados {len(key_supports)} soportes y {len(key_resistances)} resistencias clave")
        
        return {
            'supports': key_supports,
            'resistances': key_resistances
        }
    
    def zone_break_probability(self, price: Decimal, zone: Decimal, 
                             momentum: Decimal, volume_factor: Decimal = 1) -> Decimal:
        """
        Calcula la probabilidad de ruptura de una zona.
        
        Args:
            price: Precio actual
            zone: Nivel de zona
            momentum: Momentum actual
            volume_factor: Factor de volumen (1 = normal)
            
        Returns:
            Probabilidad de ruptura (0-100)
        """
        if zone == 0:
            return Decimal('50')  # Neutral
            
        # Distancia a la zona
        distance_pct = abs(price - zone) / zone
        
        # Factores base
        distance_factor = max(0, Decimal('100') * (1 - distance_pct * 100))
        momentum_factor = abs(momentum) * 2  # Amplificar momentum
        volume_factor_adjusted = min(volume_factor * 20, 30)  # Max 30 pts
        
        # Fuerza histórica de la zona (resistencia a rupturas)
        zone_strength = self.get_zone_strength(zone)
        resistance_factor = zone_strength * Decimal('0.5')  # Reduce probabilidad
        
        # Probabilidad total
        break_probability = max(0, min(100, 
            distance_factor + momentum_factor + volume_factor_adjusted - resistance_factor
        ))
        
        return break_probability