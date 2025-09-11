"""
Tests para indicadores técnicos y personalizados.
Prueba cálculos, precisión y casos edge de los indicadores.
"""

import unittest
import sys
import os
from decimal import Decimal, getcontext
import math
from typing import List

# Configurar precisión decimal
getcontext().prec = 28

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.indicators.technical import TechnicalIndicators
from src.indicators.custom import CustomIndicators


class TestTechnicalIndicators(unittest.TestCase):
    """Tests para indicadores técnicos estándar."""
    
    def setUp(self):
        """Configuración inicial."""
        self.indicators = TechnicalIndicators()
        
        # Datos de prueba - Serie alcista
        self.bullish_data = [
            Decimal('100.0'), Decimal('101.0'), Decimal('102.5'), Decimal('103.0'),
            Decimal('104.2'), Decimal('105.1'), Decimal('106.3'), Decimal('107.0'),
            Decimal('108.5'), Decimal('109.2'), Decimal('110.1'), Decimal('111.5'),
            Decimal('112.3'), Decimal('113.8'), Decimal('114.5'), Decimal('115.2')
        ]
        
        # Datos de prueba - Serie bajista
        self.bearish_data = [
            Decimal('115.0'), Decimal('114.2'), Decimal('113.1'), Decimal('112.5'),
            Decimal('111.3'), Decimal('110.8'), Decimal('109.2'), Decimal('108.7'),
            Decimal('107.4'), Decimal('106.9'), Decimal('105.5'), Decimal('104.1'),
            Decimal('103.6'), Decimal('102.3'), Decimal('101.8'), Decimal('100.5')
        ]
        
        # Datos de prueba - Serie lateral
        self.sideways_data = [
            Decimal('100.0'), Decimal('100.5'), Decimal('99.8'), Decimal('100.2'),
            Decimal('99.9'), Decimal('100.3'), Decimal('100.1'), Decimal('99.7'),
            Decimal('100.4'), Decimal('100.0'), Decimal('99.6'), Decimal('100.2'),
            Decimal('100.1'), Decimal('99.9'), Decimal('100.0'), Decimal('100.1')
        ]
        
        # Datos de velas para tests más complejos
        self.candle_data = self._generate_candle_data()
    
    def _generate_candle_data(self) -> List[dict]:
        """Genera datos de velas para testing."""
        candles = []
        base_price = Decimal('1.1500')
        
        for i in range(50):
            open_price = base_price + Decimal(str(i * 0.0001))
            close_price = open_price + Decimal(str((i % 5 - 2) * 0.0002))  # Variación
            high_price = max(open_price, close_price) + Decimal('0.0005')
            low_price = min(open_price, close_price) - Decimal('0.0003')
            
            candle = {
                'timestamp': 1609459200 + (i * 300),  # 5 min intervals
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            }
            candles.append(candle)
        
        return candles
    
    def test_sma_calculation(self):
        """Test del cálculo de Media Móvil Simple."""
        # Test básico
        sma = self.indicators.calculate_sma(self.bullish_data[:10], period=5)
        expected = sum(self.bullish_data[5:10]) / 5
        self.assertAlmostEqual(float(sma), float(expected), places=6)
        
        # Test con datos insuficientes
        sma_insufficient = self.indicators.calculate_sma(self.bullish_data[:3], period=5)
        self.assertIsNone(sma_insufficient)
        
        # Test con período igual al tamaño de datos
        sma_exact = self.indicators.calculate_sma(self.bullish_data[:5], period=5)
        expected_exact = sum(self.bullish_data[:5]) / 5
        self.assertAlmostEqual(float(sma_exact), float(expected_exact), places=6)
    
    def test_ema_calculation(self):
        """Test del cálculo de Media Móvil Exponencial."""
        # Test básico
        ema = self.indicators.calculate_ema(self.bullish_data, period=5)
        self.assertIsInstance(ema, Decimal)
        self.assertGreater(ema, 0)
        
        # EMA debería ser más responsiva que SMA
        sma = self.indicators.calculate_sma(self.bullish_data, period=5)
        # En tendencia alcista, EMA debería ser mayor que SMA
        self.assertGreater(ema, sma)
        
        # Test con datos insuficientes
        ema_insufficient = self.indicators.calculate_ema(self.bullish_data[:3], period=5)
        self.assertIsNone(ema_insufficient)
    
    def test_rsi_calculation(self):
        """Test del cálculo de RSI."""
        # Test con tendencia alcista - RSI debería estar alto
        rsi_bull = self.indicators.calculate_rsi(self.bullish_data, period=14)
        self.assertIsInstance(rsi_bull, Decimal)
        self.assertTrue(50 < rsi_bull <= 100)  # RSI alto en tendencia alcista
        
        # Test con tendencia bajista - RSI debería estar bajo
        rsi_bear = self.indicators.calculate_rsi(self.bearish_data, period=14)
        self.assertTrue(0 <= rsi_bear < 50)  # RSI bajo en tendencia bajista
        
        # Test con mercado lateral - RSI cerca de 50
        rsi_sideways = self.indicators.calculate_rsi(self.sideways_data, period=14)
        self.assertTrue(40 <= rsi_sideways <= 60)  # RSI neutro en lateral
        
        # Test con datos insuficientes
        rsi_insufficient = self.indicators.calculate_rsi(self.bullish_data[:10], period=14)
        self.assertIsNone(rsi_insufficient)
    
    def test_macd_calculation(self):
        """Test del cálculo de MACD."""
        macd_line, signal_line, histogram = self.indicators.calculate_macd(
            self.bullish_data, fast=12, slow=26, signal=9
        )
        
        # Todos los componentes deberían existir
        self.assertIsInstance(macd_line, Decimal)
        self.assertIsInstance(signal_line, Decimal)
        self.assertIsInstance(histogram, Decimal)
        
        # Histogram = MACD - Signal
        calculated_histogram = macd_line - signal_line
        self.assertAlmostEqual(float(histogram), float(calculated_histogram), places=6)
        
        # Test con datos insuficientes
        result = self.indicators.calculate_macd(self.bullish_data[:20], fast=12, slow=26, signal=9)
        self.assertEqual(result, (None, None, None))
    
    def test_bollinger_bands(self):
        """Test del cálculo de Bandas de Bollinger."""
        upper, middle, lower = self.indicators.calculate_bollinger_bands(
            self.bullish_data, period=10, std_dev=2
        )
        
        # Todas las bandas deberían existir
        self.assertIsInstance(upper, Decimal)
        self.assertIsInstance(middle, Decimal)
        self.assertIsInstance(lower, Decimal)
        
        # Orden correcto: Upper > Middle > Lower
        self.assertGreater(upper, middle)
        self.assertGreater(middle, lower)
        
        # Middle debería ser SMA
        expected_middle = self.indicators.calculate_sma(self.bullish_data, period=10)
        self.assertAlmostEqual(float(middle), float(expected_middle), places=6)
        
        # Test con diferentes desviaciones estándar
        upper_1, _, lower_1 = self.indicators.calculate_bollinger_bands(
            self.bullish_data, period=10, std_dev=1
        )
        upper_3, _, lower_3 = self.indicators.calculate_bollinger_bands(
            self.bullish_data, period=10, std_dev=3
        )
        
        # Bandas más anchas con mayor desviación
        self.assertGreater(upper_3 - lower_3, upper_1 - lower_1)
    
    def test_stochastic_oscillator(self):
        """Test del cálculo de Oscilador Estocástico."""
        if not hasattr(self.indicators, 'calculate_stochastic'):
            self.skipTest("Stochastic oscillator not implemented")
        
        k_percent, d_percent = self.indicators.calculate_stochastic(
            self.candle_data, period=14, smooth_k=3, smooth_d=3
        )
        
        # Ambos valores deberían estar entre 0 y 100
        self.assertTrue(0 <= k_percent <= 100)
        self.assertTrue(0 <= d_percent <= 100)
    
    def test_atr_calculation(self):
        """Test del cálculo de Average True Range."""
        atr = self.indicators.calculate_atr(self.candle_data, period=14)
        
        self.assertIsInstance(atr, Decimal)
        self.assertGreater(atr, 0)  # ATR siempre positivo
        
        # Test con datos insuficientes
        atr_insufficient = self.indicators.calculate_atr(self.candle_data[:10], period=14)
        self.assertIsNone(atr_insufficient)


class TestCustomIndicators(unittest.TestCase):
    """Tests para indicadores personalizados."""
    
    def setUp(self):
        """Configuración inicial."""
        self.indicators = CustomIndicators()
        
        # Generar datos de velas más realistas
        self.realistic_candles = self._generate_realistic_candles()
    
    def _generate_realistic_candles(self) -> List[dict]:
        """Genera velas con patrones más realistas."""
        candles = []
        base_price = Decimal('1.1500')
        
        # Simular diferentes fases del mercado
        phases = [
            {'trend': 0.0002, 'volatility': 0.0005, 'count': 20},  # Alcista
            {'trend': -0.0001, 'volatility': 0.0003, 'count': 15}, # Bajista ligero
            {'trend': 0.0000, 'volatility': 0.0002, 'count': 15},  # Lateral
        ]
        
        timestamp = 1609459200
        current_price = base_price
        
        for phase in phases:
            for i in range(phase['count']):
                # Simular apertura
                open_price = current_price
                
                # Simular movimiento de tendencia + ruido
                trend_move = Decimal(str(phase['trend']))
                volatility = Decimal(str(phase['volatility']))
                
                # Usar seno para simular movimiento más natural
                noise_factor = Decimal(str(math.sin(i * 0.3) * float(volatility)))
                close_price = open_price + trend_move + noise_factor
                
                # High y Low basados en volatilidad
                high_price = max(open_price, close_price) + volatility * Decimal('0.8')
                low_price = min(open_price, close_price) - volatility * Decimal('0.6')
                
                candle = {
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price
                }
                
                candles.append(candle)
                timestamp += 300
                current_price = close_price
        
        return candles
    
    def test_support_resistance_detection(self):
        """Test de detección de soportes y resistencias."""
        supports, resistances = self.indicators.detect_support_resistance(
            self.realistic_candles, lookback=10, min_touches=2
        )
        
        self.assertIsInstance(supports, list)
        self.assertIsInstance(resistances, list)
        
        # Deberían detectarse algunos niveles
        total_levels = len(supports) + len(resistances)
        self.assertGreater(total_levels, 0)
        
        # Los soportes deberían ser menores que las resistencias
        if supports and resistances:
            max_support = max(supports)
            min_resistance = min(resistances)
            self.assertLess(max_support, min_resistance)
    
    def test_trend_strength_calculation(self):
        """Test de cálculo de fuerza de tendencia."""
        strength = self.indicators.calculate_trend_strength(self.realistic_candles)
        
        self.assertIsInstance(strength, Decimal)
        self.assertTrue(-100 <= strength <= 100)
        
        # Test con tendencia claramente alcista
        bullish_candles = []
        base_price = Decimal('1.1500')
        
        for i in range(20):
            price = base_price + Decimal(str(i * 0.001))  # Tendencia fuerte
            candle = {
                'timestamp': 1609459200 + (i * 300),
                'open': price,
                'high': price + Decimal('0.0005'),
                'low': price - Decimal('0.0002'),
                'close': price + Decimal('0.0003')
            }
            bullish_candles.append(candle)
        
        bullish_strength = self.indicators.calculate_trend_strength(bullish_candles)
        self.assertGreater(bullish_strength, 50)  # Debería ser fuertemente alcista
    
    def test_volatility_calculation(self):
        """Test de cálculo de volatilidad."""
        volatility = self.indicators.calculate_volatility(self.realistic_candles, period=20)
        
        self.assertIsInstance(volatility, Decimal)
        self.assertGreaterEqual(volatility, 0)
        
        # Test con datos de alta volatilidad
        high_vol_candles = []
        base_price = Decimal('1.1500')
        
        for i in range(30):
            # Crear movimientos erráticos
            price_change = Decimal(str((i % 4 - 1.5) * 0.002))  # Cambios grandes
            close_price = base_price + price_change
            
            candle = {
                'timestamp': 1609459200 + (i * 300),
                'open': base_price,
                'high': max(base_price, close_price) + Decimal('0.001'),
                'low': min(base_price, close_price) - Decimal('0.001'),
                'close': close_price
            }
            high_vol_candles.append(candle)
            base_price = close_price
        
        high_volatility = self.indicators.calculate_volatility(high_vol_candles, period=20)
        
        # La volatilidad alta debería ser mayor que la normal
        self.assertGreater(high_volatility, volatility)
    
    def test_confluence_score_calculation(self):
        """Test de cálculo de score de confluencia."""
        # Test con todos los factores positivos
        all_positive = {
            'trend_alignment': True,
            'zone_proximity': True,
            'pattern_confirmation': True,
            'momentum_divergence': False,
            'volume_confirmation': True
        }
        
        score_high = self.indicators.calculate_confluence_score(all_positive)
        self.assertIsInstance(score_high, int)
        self.assertGreater(score_high, 60)
        
        # Test con factores mixtos
        mixed_factors = {
            'trend_alignment': True,
            'zone_proximity': False,
            'pattern_confirmation': True,
            'momentum_divergence': True,
            'volume_confirmation': False
        }
        
        score_medium = self.indicators.calculate_confluence_score(mixed_factors)
        self.assertTrue(30 <= score_medium <= 70)
        
        # Test con factores mayormente negativos
        mostly_negative = {
            'trend_alignment': False,
            'zone_proximity': False,
            'pattern_confirmation': False,
            'momentum_divergence': True,
            'volume_confirmation': False
        }
        
        score_low = self.indicators.calculate_confluence_score(mostly_negative)
        self.assertLess(score_low, 40)
    
    def test_price_action_patterns(self):
        """Test de detección de patrones de price action."""
        # Test de detección de martillo
        hammer_candles = [
            {
                'open': Decimal('1.1500'),
                'high': Decimal('1.1505'),
                'low': Decimal('1.1450'),  # Mecha inferior larga
                'close': Decimal('1.1498')
            },
            {
                'open': Decimal('1.1498'),
                'high': Decimal('1.1510'),
                'low': Decimal('1.1495'),
                'close': Decimal('1.1508')  # Confirmación alcista
            }
        ]
        
        is_hammer = self.indicators.detect_hammer_pattern(hammer_candles)
        self.assertIsInstance(is_hammer, bool)
        
        # Test de detección de doji
        doji_candle = {
            'open': Decimal('1.1500'),
            'high': Decimal('1.1510'),
            'low': Decimal('1.1490'),
            'close': Decimal('1.1501')  # Casi igual al open
        }
        
        is_doji = self.indicators.detect_doji_pattern([doji_candle])
        self.assertIsInstance(is_doji, bool)
    
    def test_momentum_indicators(self):
        """Test de indicadores de momentum personalizados."""
        momentum = self.indicators.calculate_momentum(self.realistic_candles, period=10)
        
        self.assertIsInstance(momentum, Decimal)
        
        # Momentum debería reflejar la dirección del precio
        price_change = self.realistic_candles[-1]['close'] - self.realistic_candles[-11]['close']
        
        if price_change > 0:
            self.assertGreater(momentum, 0)
        elif price_change < 0:
            self.assertLess(momentum, 0)
    
    def test_edge_cases(self):
        """Test de casos límite y manejo de errores."""
        # Datos vacíos
        empty_result = self.indicators.calculate_trend_strength([])
        self.assertEqual(empty_result, Decimal('0'))
        
        # Datos con un solo elemento
        single_candle = [self.realistic_candles[0]]
        single_result = self.indicators.calculate_volatility(single_candle, period=5)
        self.assertEqual(single_result, Decimal('0'))
        
        # Datos con precios idénticos (sin volatilidad)
        flat_candles = []
        for i in range(20):
            candle = {
                'timestamp': 1609459200 + (i * 300),
                'open': Decimal('1.1500'),
                'high': Decimal('1.1500'),
                'low': Decimal('1.1500'),
                'close': Decimal('1.1500')
            }
            flat_candles.append(candle)
        
        flat_volatility = self.indicators.calculate_volatility(flat_candles, period=10)
        self.assertEqual(flat_volatility, Decimal('0'))


class TestIndicatorAccuracy(unittest.TestCase):
    """Tests de precisión y validación matemática."""
    
    def setUp(self):
        """Configuración para tests de precisión."""
        self.indicators = TechnicalIndicators()
        
        # Datos de referencia conocidos
        self.reference_data = [
            Decimal('44.34'), Decimal('44.09'), Decimal('44.15'), Decimal('43.61'),
            Decimal('44.33'), Decimal('44.83'), Decimal('45.85'), Decimal('47.25'),
            Decimal('49.21'), Decimal('47.04'), Decimal('46.09'), Decimal('48.67'),
            Decimal('46.09'), Decimal('47.69'), Decimal('47.00'), Decimal('44.25')
        ]
    
    def test_sma_mathematical_accuracy(self):
        """Test de precisión matemática del SMA."""
        # Calcular SMA manualmente
        period = 5
        manual_sma = sum(self.reference_data[-period:]) / period
        
        # Calcular con el indicador
        calculated_sma = self.indicators.calculate_sma(self.reference_data, period)
        
        # Deberían ser idénticos
        self.assertEqual(calculated_sma, manual_sma)
    
    def test_ema_convergence(self):
        """Test de convergencia del EMA."""
        # Con período largo, EMA debería aproximarse a SMA
        long_data = [Decimal('100')] * 100  # Datos constantes
        
        ema = self.indicators.calculate_ema(long_data, period=50)
        sma = self.indicators.calculate_sma(long_data, period=50)
        
        # Deberían ser muy cercanos con datos constantes
        difference = abs(ema - sma)
        self.assertLess(difference, Decimal('0.01'))
    
    def test_rsi_boundary_conditions(self):
        """Test de condiciones límite del RSI."""
        # Datos siempre crecientes - RSI debería tender a 100
        always_up = [Decimal(str(i)) for i in range(1, 21)]
        rsi_up = self.indicators.calculate_rsi(always_up, period=14)
        self.assertGreater(rsi_up, 70)  # Debería estar en zona de sobrecompra
        
        # Datos siempre decrecientes - RSI debería tender a 0
        always_down = [Decimal(str(20 - i)) for i in range(20)]
        rsi_down = self.indicators.calculate_rsi(always_down, period=14)
        self.assertLess(rsi_down, 30)  # Debería estar en zona de sobreventa


class TestIndicatorPerformance(unittest.TestCase):
    """Tests de rendimiento de indicadores."""
    
    def setUp(self):
        """Configuración para tests de rendimiento."""
        self.indicators = TechnicalIndicators()
        self.custom_indicators = CustomIndicators()
        
        # Generar dataset grande para pruebas de rendimiento
        self.large_dataset = self._generate_large_dataset(10000)
    
    def _generate_large_dataset(self, size: int) -> List[dict]:
        """Genera un dataset grande para tests de performance."""
        import random
        
        candles = []
        base_price = Decimal('1.1500')
        timestamp = 1609459200
        
        for i in range(size):
            # Movimiento aleatorio pero realista
            change = Decimal(str(random.uniform(-0.0020, 0.0020)))
            close_price = max(base_price + change, Decimal('0.0001'))  # Evitar precios negativos
            
            high_price = close_price + Decimal(str(random.uniform(0, 0.0010)))
            low_price = close_price - Decimal(str(random.uniform(0, 0.0010)))
            open_price = base_price
            
            candle = {
                'timestamp': timestamp + (i * 60),  # 1 minuto
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            }
            
            candles.append(candle)
            base_price = close_price
        
        return candles
    
    def test_sma_performance(self):
        """Test de rendimiento del SMA."""
        import time
        
        prices = [candle['close'] for candle in self.large_dataset]
        
        start_time = time.time()
        
        # Calcular SMA múltiples veces
        for _ in range(100):
            self.indicators.calculate_sma(prices, period=20)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Debería ser muy rápido (< 1ms por cálculo)
        self.assertLess(avg_time, 0.001)
    
    def test_large_dataset_memory_usage(self):
        """Test de uso de memoria con datasets grandes."""
        import tracemalloc
        
        tracemalloc.start()
        
        # Procesar dataset grande
        prices = [candle['close'] for candle in self.large_dataset]
        
        # Calcular múltiples indicadores
        self.indicators.calculate_sma(prices, period=50)
        self.indicators.calculate_ema(prices, period=50)
        self.indicators.calculate_rsi(prices, period=14)
        
        # Indicadores personalizados
        self.custom_indicators.calculate_volatility(self.large_dataset[:1000], period=20)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # El uso de memoria no debería ser excesivo (menos de 100MB)
        peak_mb = peak / 1024 / 1024
        self.assertLess(peak_mb, 100)


class TestIndicatorIntegration(unittest.TestCase):
    """Tests de integración entre diferentes indicadores."""
    
    def setUp(self):
        """Configuración para tests de integración."""
        self.technical = TechnicalIndicators()
        self.custom = CustomIndicators()
        
        # Datos de mercado realistas
        self.market_data = self._create_market_scenario()
    
    def _create_market_scenario(self) -> List[dict]:
        """Crea un escenario de mercado con diferentes fases."""
        import math
        
        candles = []
        base_price = Decimal('1.1500')
        timestamp = 1609459200
        
        # Fase 1: Tendencia alcista
        for i in range(50):
            trend = Decimal(str(i * 0.0002))
            noise = Decimal(str(math.sin(i * 0.2) * 0.0005))
            close_price = base_price + trend + noise
            
            candle = {
                'timestamp': timestamp + (i * 300),
                'open': base_price + trend,
                'high': close_price + Decimal('0.0003'),
                'low': close_price - Decimal('0.0002'),
                'close': close_price
            }
            candles.append(candle)
        
        # Fase 2: Consolidación
        consolidation_price = close_price
        for i in range(30):
            noise = Decimal(str(math.sin(i * 0.5) * 0.0003))
            close_price = consolidation_price + noise
            
            candle = {
                'timestamp': timestamp + ((50 + i) * 300),
                'open': consolidation_price,
                'high': close_price + Decimal('0.0002'),
                'low': close_price - Decimal('0.0002'),
                'close': close_price
            }
            candles.append(candle)
        
        return candles
    
    def test_trend_confirmation_across_indicators(self):
        """Test de confirmación de tendencia entre múltiples indicadores."""
        prices = [candle['close'] for candle in self.market_data]
        
        # Calcular diferentes indicadores
        ema_fast = self.technical.calculate_ema(prices, period=8)
        ema_slow = self.technical.calculate_ema(prices, period=21)
        rsi = self.technical.calculate_rsi(prices, period=14)
        
        # Calcular indicadores personalizados
        trend_strength = self.custom.calculate_trend_strength(self.market_data)
        
        # En la primera fase (alcista), todos deberían coincidir
        if ema_fast and ema_slow:
            ema_bullish = ema_fast > ema_slow
            rsi_bullish = rsi > 50
            trend_bullish = trend_strength > 0
            
            # La mayoría debería indicar tendencia alcista
            bullish_count = sum([ema_bullish, rsi_bullish, trend_bullish])
            self.assertGreaterEqual(bullish_count, 2)
    
    def test_volatility_indicator_consistency(self):
        """Test de consistencia entre indicadores de volatilidad."""
        # ATR del indicador técnico
        atr = self.technical.calculate_atr(self.market_data, period=14)
        
        # Volatilidad del indicador personalizado
        volatility = self.custom.calculate_volatility(self.market_data, period=14)
        
        # Ambos deberían indicar niveles similares de volatilidad
        if atr and volatility:
            # Normalizar para comparar (ambos deberían moverse en la misma dirección)
            self.assertGreater(atr, 0)
            self.assertGreater(volatility, 0)


def run_comprehensive_tests():
    """Ejecuta todos los tests de forma comprensiva."""
    test_classes = [
        TestTechnicalIndicators,
        TestCustomIndicators,
        TestIndicatorAccuracy,
        TestIndicatorPerformance,
        TestIndicatorIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


if __name__ == '__main__':
    import logging
    
    # Configurar logging menos verbose para tests
    logging.basicConfig(level=logging.WARNING)
    
    print("Ejecutando tests de indicadores...")
    print("=" * 60)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = run_comprehensive_tests()
    result = runner.run(suite)
    
    # Resumen de resultados
    print("\n" + "=" * 60)
    print("RESUMEN DE TESTS DE INDICADORES")
    print("=" * 60)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Éxitos: {result.testsRun - len(result.errors) - len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    
    if result.errors:
        print("\nERRORES ENCONTRADOS:")
        for test, error in result.errors:
            print(f"- {test}: {error.split('\\n')[-2] if '\\n' in error else error}")
    
    if result.failures:
        print("\nFALLOS ENCONTRADOS:")
        for test, failure in result.failures:
            print(f"- {test}: {failure.split('\\n')[-2] if '\\n' in failure else failure}")
    
    # Determinar código de salida
    success = result.wasSuccessful()
    print(f"\nResultado final: {'✅ TODOS LOS TESTS PASARON' if success else '❌ ALGUNOS TESTS FALLARON'}")
    
    exit(0 if success else 1)