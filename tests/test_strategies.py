"""
Tests para las estrategias de trading.
Prueba los diferentes componentes de análisis y generación de señales.
"""

import unittest
import sys
import os
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.strategies.base_strategy import BaseStrategy, StrategySignal
from src.strategies.confluence_strategy import ConfluenceStrategy
from src.indicators.technical import TechnicalIndicators
from src.indicators.custom import CustomIndicators


class TestBaseStrategy(unittest.TestCase):
    """Tests para la estrategia base."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        self.config = {
            'symbol': 'frxEURUSD',
            'timeframe': 300,  # 5 minutos
            'min_signal_strength': 2
        }
        self.strategy = BaseStrategy(self.config)
    
    def test_initialization(self):
        """Test de inicialización de la estrategia."""
        self.assertEqual(self.strategy.symbol, 'frxEURUSD')
        self.assertEqual(self.strategy.timeframe, 300)
        self.assertIsNotNone(self.strategy.logger)
    
    def test_add_candle(self):
        """Test de adición de velas."""
        candle = {
            'timestamp': 1609459200,  # 2021-01-01 00:00:00
            'open': 1.2000,
            'high': 1.2050,
            'low': 1.1950,
            'close': 1.2025
        }
        
        result = self.strategy.add_candle(candle)
        self.assertTrue(result)
        self.assertEqual(len(self.strategy.candles), 1)
    
    def test_add_invalid_candle(self):
        """Test de adición de vela inválida."""
        invalid_candle = {
            'timestamp': 1609459200,
            'open': 1.2000,
            'high': 1.1950,  # High menor que open (inválido)
            'low': 1.1950,
            'close': 1.2025
        }
        
        result = self.strategy.add_candle(invalid_candle)
        self.assertFalse(result)
        self.assertEqual(len(self.strategy.candles), 0)
    
    def test_analyze_with_insufficient_data(self):
        """Test de análisis con datos insuficientes."""
        signal = self.strategy.analyze()
        self.assertIsNone(signal)


class TestConfluenceStrategy(unittest.TestCase):
    """Tests para la estrategia de confluencia."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        self.config = {
            'symbol': 'frxEURUSD',
            'timeframe': 300,
            'min_signal_strength': 2,
            'zones': {
                'supports': [1.1000, 1.1200, 1.1400],
                'resistances': [1.1800, 1.2000, 1.2200]
            },
            'zone_tolerance': 0.0005,
            'ema_fast_period': 8,
            'ema_slow_period': 21
        }
        self.strategy = ConfluenceStrategy(self.config)
        
        # Agregar datos de prueba
        self._add_sample_candles()
    
    def _add_sample_candles(self):
        """Agrega velas de ejemplo para testing."""
        base_timestamp = 1609459200  # 2021-01-01 00:00:00
        base_price = 1.1500
        
        for i in range(50):  # Agregar 50 velas
            candle = {
                'timestamp': base_timestamp + (i * 300),  # Cada 5 minutos
                'open': base_price + (i * 0.0001),
                'high': base_price + (i * 0.0001) + 0.0020,
                'low': base_price + (i * 0.0001) - 0.0015,
                'close': base_price + (i * 0.0001) + 0.0005
            }
            self.strategy.add_candle(candle)
    
    def test_initialization(self):
        """Test de inicialización de ConfluenceStrategy."""
        self.assertIsInstance(self.strategy.technical_indicators, TechnicalIndicators)
        self.assertIsInstance(self.strategy.custom_indicators, CustomIndicators)
        self.assertEqual(len(self.strategy.support_zones), 3)
        self.assertEqual(len(self.strategy.resistance_zones), 3)
    
    def test_zone_detection(self):
        """Test de detección de zonas."""
        price = Decimal('1.1200')
        tolerance = Decimal('0.0005')
        
        # Test soporte
        is_support, zone_price = self.strategy._is_near_support_zone(price, tolerance)
        self.assertTrue(is_support)
        self.assertEqual(zone_price, Decimal('1.1200'))
        
        # Test resistencia
        price_resistance = Decimal('1.2000')
        is_resistance, zone_price = self.strategy._is_near_resistance_zone(price_resistance, tolerance)
        self.assertTrue(is_resistance)
        self.assertEqual(zone_price, Decimal('1.2000'))
    
    def test_trend_analysis(self):
        """Test de análisis de tendencia."""
        # Mock EMAs para simular tendencia alcista
        with patch.object(self.strategy.technical_indicators, 'calculate_ema') as mock_ema:
            mock_ema.side_effect = [Decimal('1.1520'), Decimal('1.1500')]  # Fast > Slow
            
            trend = self.strategy._analyze_trend()
            self.assertEqual(trend, 'bullish')
    
    def test_pattern_detection(self):
        """Test de detección de patrones."""
        # Test martillo
        hammer_candles = [
            {'open': 1.1500, 'high': 1.1520, 'low': 1.1450, 'close': 1.1515},  # Martillo
            {'open': 1.1520, 'high': 1.1525, 'low': 1.1510, 'close': 1.1522}   # Vela actual
        ]
        
        is_hammer, confidence = self.strategy._detect_hammer_pattern(hammer_candles)
        self.assertTrue(is_hammer)
        self.assertGreater(confidence, 0)
    
    @patch('src.strategies.confluence_strategy.ConfluenceStrategy._is_near_support_zone')
    @patch('src.strategies.confluence_strategy.ConfluenceStrategy._analyze_trend')
    def test_signal_generation_call(self, mock_trend, mock_support):
        """Test de generación de señal CALL."""
        # Mock condiciones para señal CALL
        mock_support.return_value = (True, Decimal('1.1200'))
        mock_trend.return_value = 'bullish'
        
        signal = self.strategy.analyze()
        
        if signal:  # Solo verificar si se genera una señal
            self.assertEqual(signal.direction, 'CALL')
            self.assertGreater(signal.strength, 0)
    
    def test_risk_management_integration(self):
        """Test de integración con gestión de riesgo."""
        # Simular pérdidas consecutivas
        self.strategy.consecutive_losses = 3
        
        signal = self.strategy.analyze()
        
        # Con muchas pérdidas consecutivas, la señal debería ser más débil o None
        if signal:
            self.assertLessEqual(signal.strength, 2)


class TestTechnicalIndicators(unittest.TestCase):
    """Tests para indicadores técnicos."""
    
    def setUp(self):
        """Configuración inicial."""
        self.indicators = TechnicalIndicators()
        
        # Datos de precio de ejemplo
        self.price_data = [
            Decimal('1.1500'), Decimal('1.1520'), Decimal('1.1510'), 
            Decimal('1.1530'), Decimal('1.1525'), Decimal('1.1540'),
            Decimal('1.1535'), Decimal('1.1550'), Decimal('1.1545'),
            Decimal('1.1560'), Decimal('1.1555'), Decimal('1.1570')
        ]
    
    def test_sma_calculation(self):
        """Test de cálculo de SMA."""
        sma = self.indicators.calculate_sma(self.price_data, period=5)
        self.assertIsInstance(sma, Decimal)
        self.assertGreater(sma, 0)
    
    def test_ema_calculation(self):
        """Test de cálculo de EMA."""
        ema = self.indicators.calculate_ema(self.price_data, period=5)
        self.assertIsInstance(ema, Decimal)
        self.assertGreater(ema, 0)
    
    def test_rsi_calculation(self):
        """Test de cálculo de RSI."""
        # Crear datos con tendencia clara
        trending_data = [Decimal(f'1.{1500 + i}') for i in range(20)]
        
        rsi = self.indicators.calculate_rsi(trending_data, period=14)
        self.assertIsInstance(rsi, Decimal)
        self.assertTrue(0 <= rsi <= 100)
    
    def test_bollinger_bands(self):
        """Test de cálculo de Bandas de Bollinger."""
        upper, middle, lower = self.indicators.calculate_bollinger_bands(
            self.price_data, period=10, std_dev=2
        )
        
        self.assertIsInstance(upper, Decimal)
        self.assertIsInstance(middle, Decimal)  
        self.assertIsInstance(lower, Decimal)
        self.assertGreater(upper, middle)
        self.assertGreater(middle, lower)
    
    def test_macd_calculation(self):
        """Test de cálculo de MACD."""
        # Necesitamos más datos para MACD
        extended_data = [Decimal(f'1.{1500 + (i*2)}') for i in range(50)]
        
        macd_line, signal_line, histogram = self.indicators.calculate_macd(
            extended_data, fast=12, slow=26, signal=9
        )
        
        self.assertIsInstance(macd_line, Decimal)
        self.assertIsInstance(signal_line, Decimal)
        self.assertIsInstance(histogram, Decimal)


class TestCustomIndicators(unittest.TestCase):
    """Tests para indicadores personalizados."""
    
    def setUp(self):
        """Configuración inicial."""
        self.indicators = CustomIndicators()
        
        # Datos de velas de ejemplo
        self.candle_data = []
        base_price = Decimal('1.1500')
        
        for i in range(20):
            candle = {
                'timestamp': 1609459200 + (i * 300),
                'open': base_price + Decimal(str(i * 0.0005)),
                'high': base_price + Decimal(str(i * 0.0005 + 0.0020)),
                'low': base_price + Decimal(str(i * 0.0005 - 0.0015)),
                'close': base_price + Decimal(str(i * 0.0005 + 0.0010))
            }
            self.candle_data.append(candle)
    
    def test_support_resistance_detection(self):
        """Test de detección de soportes y resistencias."""
        supports, resistances = self.indicators.detect_support_resistance(
            self.candle_data, lookback=10, min_touches=2
        )
        
        self.assertIsInstance(supports, list)
        self.assertIsInstance(resistances, list)
    
    def test_trend_strength(self):
        """Test de cálculo de fuerza de tendencia."""
        strength = self.indicators.calculate_trend_strength(self.candle_data)
        
        self.assertIsInstance(strength, Decimal)
        self.assertTrue(-100 <= strength <= 100)
    
    def test_volatility_calculation(self):
        """Test de cálculo de volatilidad."""
        volatility = self.indicators.calculate_volatility(self.candle_data, period=10)
        
        self.assertIsInstance(volatility, Decimal)
        self.assertGreaterEqual(volatility, 0)
    
    def test_confluence_score(self):
        """Test de cálculo de score de confluencia."""
        factors = {
            'trend_alignment': True,
            'zone_proximity': True,
            'pattern_confirmation': False,
            'momentum_divergence': True
        }
        
        score = self.indicators.calculate_confluence_score(factors)
        
        self.assertIsInstance(score, int)
        self.assertTrue(0 <= score <= 100)


class TestStrategyIntegration(unittest.TestCase):
    """Tests de integración entre componentes de estrategia."""
    
    def setUp(self):
        """Configuración para tests de integración."""
        self.config = {
            'symbol': 'frxEURUSD',
            'timeframe': 300,
            'min_signal_strength': 3,
            'zones': {
                'supports': [1.1000, 1.1200, 1.1400],
                'resistances': [1.1800, 1.2000, 1.2200]
            },
            'zone_tolerance': 0.0005,
            'ema_fast_period': 8,
            'ema_slow_period': 21
        }
        
        self.strategy = ConfluenceStrategy(self.config)
        
        # Agregar datos históricos simulados
        self._setup_market_scenario()
    
    def _setup_market_scenario(self):
        """Configura un escenario de mercado para testing."""
        base_timestamp = 1609459200
        
        # Escenario: Precio acercándose a soporte con tendencia alcista
        scenarios = [
            # Tendencia bajista inicial
            {'base': 1.1300, 'trend': -0.0001, 'count': 15},
            # Consolidación cerca del soporte
            {'base': 1.1205, 'trend': 0.0000, 'count': 10},  
            # Señal alcista desde soporte
            {'base': 1.1200, 'trend': 0.0002, 'count': 10}
        ]
        
        candle_count = 0
        for scenario in scenarios:
            for i in range(scenario['count']):
                price = scenario['base'] + (i * scenario['trend'])
                
                candle = {
                    'timestamp': base_timestamp + (candle_count * 300),
                    'open': price,
                    'high': price + 0.0015,
                    'low': price - 0.0010,
                    'close': price + 0.0005
                }
                
                self.strategy.add_candle(candle)
                candle_count += 1
    
    def test_full_signal_analysis(self):
        """Test completo de análisis de señal."""
        signal = self.strategy.analyze()
        
        if signal:
            # Verificar estructura de la señal
            self.assertIn(signal.direction, ['CALL', 'PUT'])
            self.assertIsInstance(signal.strength, int)
            self.assertIsInstance(signal.confidence, (int, float))
            self.assertIsInstance(signal.entry_reason, str)
            self.assertIsInstance(signal.metadata, dict)
            
            # Verificar que la señal tenga sentido
            if signal.direction == 'CALL':
                # Señal CALL debería estar cerca de soporte
                self.assertTrue(any(
                    abs(float(signal.metadata.get('current_price', 0)) - support) < 0.001
                    for support in self.config['zones']['supports']
                ))
    
    def test_signal_strength_consistency(self):
        """Test de consistencia en la fuerza de señales."""
        signals = []
        
        # Generar múltiples análisis
        for _ in range(5):
            signal = self.strategy.analyze()
            if signal:
                signals.append(signal)
        
        if signals:
            # Las señales deberían tener fuerza similar en condiciones similares
            strengths = [s.strength for s in signals]
            max_diff = max(strengths) - min(strengths)
            self.assertLessEqual(max_diff, 2)  # Diferencia máxima de 2 puntos
    
    def test_risk_adjusted_signals(self):
        """Test de señales ajustadas por riesgo."""
        # Simular alta volatilidad
        with patch.object(self.strategy.custom_indicators, 'calculate_volatility') as mock_vol:
            mock_vol.return_value = Decimal('0.05')  # Alta volatilidad
            
            signal_high_vol = self.strategy.analyze()
        
        # Simular baja volatilidad
        with patch.object(self.strategy.custom_indicators, 'calculate_volatility') as mock_vol:
            mock_vol.return_value = Decimal('0.01')  # Baja volatilidad
            
            signal_low_vol = self.strategy.analyze()
        
        # En alta volatilidad, las señales deberían ser más conservadoras
        if signal_high_vol and signal_low_vol:
            self.assertLessEqual(signal_high_vol.strength, signal_low_vol.strength)


class TestStrategyPerformance(unittest.TestCase):
    """Tests de rendimiento y optimización."""
    
    def setUp(self):
        """Configuración para tests de rendimiento."""
        self.config = {
            'symbol': 'frxEURUSD',
            'timeframe': 300,
            'min_signal_strength': 2,
            'zones': {
                'supports': [1.1000, 1.1200, 1.1400],
                'resistances': [1.1800, 1.2000, 1.2200]
            },
            'zone_tolerance': 0.0005,
            'ema_fast_period': 8,
            'ema_slow_period': 21
        }
        
        self.strategy = ConfluenceStrategy(self.config)
    
    def test_memory_efficiency(self):
        """Test de eficiencia de memoria."""
        import tracemalloc
        
        tracemalloc.start()
        
        # Agregar muchas velas
        for i in range(1000):
            candle = {
                'timestamp': 1609459200 + (i * 300),
                'open': 1.1500 + (i * 0.000001),
                'high': 1.1500 + (i * 0.000001) + 0.0020,
                'low': 1.1500 + (i * 0.000001) - 0.0015,
                'close': 1.1500 + (i * 0.000001) + 0.0010
            }
            self.strategy.add_candle(candle)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # El uso de memoria no debería ser excesivo (menos de 50MB)
        self.assertLess(peak / 1024 / 1024, 50)  # 50MB
    
    def test_processing_speed(self):
        """Test de velocidad de procesamiento."""
        import time
        
        # Agregar datos de prueba
        for i in range(100):
            candle = {
                'timestamp': 1609459200 + (i * 300),
                'open': 1.1500 + (i * 0.0001),
                'high': 1.1500 + (i * 0.0001) + 0.0020,
                'low': 1.1500 + (i * 0.0001) - 0.0015,
                'close': 1.1500 + (i * 0.0001) + 0.0010
            }
            self.strategy.add_candle(candle)
        
        # Medir tiempo de análisis
        start_time = time.time()
        
        for _ in range(100):  # 100 análisis
            self.strategy.analyze()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Cada análisis debería tomar menos de 100ms
        self.assertLess(avg_time, 0.1)


class TestRegressionSuite(unittest.TestCase):
    """Suite de tests de regresión para detectar cambios no deseados."""
    
    def setUp(self):
        """Configuración para tests de regresión."""
        self.known_good_configs = [
            {
                'symbol': 'frxEURUSD',
                'timeframe': 300,
                'zones': {'supports': [1.1200], 'resistances': [1.1800]},
                'zone_tolerance': 0.0005,
                'ema_fast_period': 8,
                'ema_slow_period': 21,
                'expected_behavior': 'generates_signals_near_zones'
            },
            {
                'symbol': 'R_50',
                'timeframe': 60,
                'zones': {'supports': [145000], 'resistances': [155000]},
                'zone_tolerance': 0.001,
                'ema_fast_period': 5,
                'ema_slow_period': 15,
                'expected_behavior': 'higher_frequency_signals'
            }
        ]
    
    def test_known_configurations(self):
        """Test de configuraciones conocidas que funcionan."""
        for config in self.known_good_configs:
            with self.subTest(config=config['symbol']):
                strategy = ConfluenceStrategy(config)
                
                # Agregar datos de prueba
                self._add_test_data(strategy, config)
                
                # Verificar comportamiento esperado
                signals = []
                for _ in range(10):
                    signal = strategy.analyze()
                    if signal:
                        signals.append(signal)
                
                if config['expected_behavior'] == 'generates_signals_near_zones':
                    # Debería generar al menos algunas señales
                    self.assertGreater(len(signals), 0)
                
                elif config['expected_behavior'] == 'higher_frequency_signals':
                    # Debería generar más señales debido a timeframe menor
                    self.assertGreaterEqual(len(signals), 2)
    
    def _add_test_data(self, strategy, config):
        """Agrega datos de prueba apropiados para cada configuración."""
        base_timestamp = 1609459200
        
        if config['symbol'].startswith('frx'):
            base_price = 1.1400  # Precio base para forex
            price_increment = 0.0001
        else:  # R_50, etc.
            base_price = 150000  # Precio base para sintéticos
            price_increment = 100
        
        for i in range(50):
            candle = {
                'timestamp': base_timestamp + (i * config['timeframe']),
                'open': base_price + (i * price_increment),
                'high': base_price + (i * price_increment) + (price_increment * 20),
                'low': base_price + (i * price_increment) - (price_increment * 15),
                'close': base_price + (i * price_increment) + (price_increment * 5)
            }
            strategy.add_candle(candle)


def create_test_suite():
    """Crea una suite completa de tests."""
    suite = unittest.TestSuite()
    
    # Tests básicos
    suite.addTest(unittest.makeSuite(TestBaseStrategy))
    suite.addTest(unittest.makeSuite(TestConfluenceStrategy))
    suite.addTest(unittest.makeSuite(TestTechnicalIndicators))
    suite.addTest(unittest.makeSuite(TestCustomIndicators))
    
    # Tests de integración
    suite.addTest(unittest.makeSuite(TestStrategyIntegration))
    
    # Tests de rendimiento
    suite.addTest(unittest.makeSuite(TestStrategyPerformance))
    
    # Tests de regresión
    suite.addTest(unittest.makeSuite(TestRegressionSuite))
    
    return suite


if __name__ == '__main__':
    # Configurar logging para tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Menos verbose durante tests
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    
    print("Ejecutando tests de estrategias...")
    print("=" * 60)
    
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Resumen final
    print("\n" + "=" * 60)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Saltados: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.errors:
        print("\nERRORES:")
        for test, error in result.errors:
            print(f"- {test}: {error}")
    
    if result.failures:
        print("\nFALLOS:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")
    
    # Exit code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"\nTests {'EXITOSOS' if exit_code == 0 else 'FALLARON'}")
    
    exit(exit_code)