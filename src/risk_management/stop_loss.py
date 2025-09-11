"""
Gestión de Stop Loss y Control de Riesgos Avanzado
Maneja diferentes tipos de stop loss y validaciones de riesgo.
"""

import logging
from decimal import Decimal, getcontext
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Configurar precisión decimal
getcontext().prec = 28


class StopLossType(Enum):
    """Tipos de stop loss disponibles."""
    FIXED_AMOUNT = "fixed_amount"
    PERCENTAGE = "percentage"
    TRAILING = "trailing"
    TIME_BASED = "time_based"
    ATR_BASED = "atr_based"
    EQUITY_CURVE = "equity_curve"


class RiskLevel(Enum):
    """Niveles de riesgo para el trading."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class StopLossManager:
    """Gestor avanzado de stop loss y control de riesgos."""
    
    def __init__(self, config: Dict[str, Any]):
        """Inicializa el gestor de stop loss."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuración básica
        self.max_daily_loss = Decimal(str(config.get('max_daily_loss', '50.00')))
        self.max_weekly_loss = Decimal(str(config.get('max_weekly_loss', '200.00')))
        self.max_consecutive_losses = int(config.get('max_consecutive_losses', 5))
        self.max_daily_trades = int(config.get('max_daily_trades', 20))
        
        # Configuración de stop loss
        self.stop_loss_percentage = Decimal(str(config.get('stop_loss_percentage', '0.02')))  # 2%
        self.trailing_stop_distance = Decimal(str(config.get('trailing_stop_distance', '0.01')))  # 1%
        self.time_based_stop_minutes = int(config.get('time_based_stop_minutes', 15))
        
        # Estado del sistema
        self.initial_balance = None
        self.current_balance = None
        self.daily_start_balance = None
        self.weekly_start_balance = None
        
        # Estadísticas de trading
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Historial de trades
        self.trade_history: List[Dict] = []
        self.daily_pnl_history: List[Decimal] = []
        
        # Controles de tiempo
        self.last_trade_time = None
        self.session_start_time = datetime.now()
        self.daily_reset_time = None
        
        # Estado de stop loss activo
        self.active_stops: Dict[str, Dict] = {}
        
        self.logger.info("StopLossManager inicializado")
        self.logger.info(f"Límite pérdida diaria: ${self.max_daily_loss}")
        self.logger.info(f"Límite pérdidas consecutivas: {self.max_consecutive_losses}")

    def set_initial_balance(self, balance: Decimal) -> None:
        """Establece el balance inicial para calcular pérdidas."""
        self.initial_balance = balance
        self.current_balance = balance
        self.daily_start_balance = balance
        self.weekly_start_balance = balance
        
        self.logger.info(f"Balance inicial establecido: ${balance:.2f}")

    def update_balance(self, new_balance: Decimal) -> None:
        """Actualiza el balance actual."""
        if self.current_balance is not None:
            change = new_balance - self.current_balance
            self.logger.debug(f"Balance actualizado: ${self.current_balance:.2f} -> ${new_balance:.2f} ({change:+.2f})")
        
        self.current_balance = new_balance

    def should_stop_trading(self) -> Tuple[bool, str]:
        """Evalúa si se debe detener el trading por límites de riesgo."""
        
        # Verificar balance disponible
        if not self._validate_balances():
            return True, "Datos de balance inválidos"
        
        # 1. Verificar límite de pérdida diaria
        if self._check_daily_loss_limit():
            daily_loss = self.daily_start_balance - self.current_balance
            return True, f"Límite de pérdida diaria alcanzado: ${daily_loss:.2f}"
        
        # 2. Verificar límite de pérdida semanal
        if self._check_weekly_loss_limit():
            weekly_loss = self.weekly_start_balance - self.current_balance
            return True, f"Límite de pérdida semanal alcanzado: ${weekly_loss:.2f}"
        
        # 3. Verificar pérdidas consecutivas
        if self.consecutive_losses >= self.max_consecutive_losses:
            return True, f"Límite de pérdidas consecutivas: {self.consecutive_losses}"
        
        # 4. Verificar límite de trades diarios
        if self.daily_trades >= self.max_daily_trades:
            return True, f"Límite de trades diarios alcanzado: {self.daily_trades}"
        
        # 5. Verificar drawdown extremo (más del 20% del balance inicial)
        if self.initial_balance and self.current_balance:
            drawdown_pct = ((self.initial_balance - self.current_balance) / self.initial_balance) * 100
            if drawdown_pct > 20:
                return True, f"Drawdown extremo detectado: {drawdown_pct:.1f}%"
        
        return False, ""

    def _validate_balances(self) -> bool:
        """Valida que los balances estén configurados correctamente."""
        return (self.initial_balance is not None and 
                self.current_balance is not None and 
                self.daily_start_balance is not None)

    def _check_daily_loss_limit(self) -> bool:
        """Verifica el límite de pérdida diaria."""
        if not self.daily_start_balance or not self.current_balance:
            return False
        
        daily_loss = self.daily_start_balance - self.current_balance
        return daily_loss >= self.max_daily_loss

    def _check_weekly_loss_limit(self) -> bool:
        """Verifica el límite de pérdida semanal."""
        if not self.weekly_start_balance or not self.current_balance:
            return False
        
        weekly_loss = self.weekly_start_balance - self.current_balance
        return weekly_loss >= self.max_weekly_loss

    def calculate_position_size(self, signal_strength: int = 1, 
                              current_risk_level: RiskLevel = RiskLevel.MEDIUM) -> Decimal:
        """Calcula el tamaño de posición basado en el riesgo y la fuerza de la señal."""
        if not self.current_balance:
            return Decimal('1.0')  # Valor por defecto
        
        # Tamaño base como porcentaje del balance
        base_risk_pct = {
            RiskLevel.LOW: Decimal('0.01'),      # 1%
            RiskLevel.MEDIUM: Decimal('0.02'),   # 2% 
            RiskLevel.HIGH: Decimal('0.03'),     # 3%
            RiskLevel.EXTREME: Decimal('0.05')   # 5%
        }
        
        risk_pct = base_risk_pct.get(current_risk_level, Decimal('0.02'))
        
        # Ajustar por fuerza de la señal (1-5)
        signal_multiplier = Decimal(str(min(max(signal_strength, 1), 5))) / Decimal('3')
        
        # Ajustar por performance reciente
        performance_multiplier = self._get_performance_multiplier()
        
        # Calcular tamaño final
        position_size = self.current_balance * risk_pct * signal_multiplier * performance_multiplier
        
        # Límites mínimos y máximos
        min_size = Decimal('1.0')
        max_size = self.current_balance * Decimal('0.1')  # Máximo 10% del balance
        
        position_size = max(min_size, min(position_size, max_size))
        
        self.logger.debug(f"Tamaño de posición calculado: ${position_size:.2f}")
        self.logger.debug(f"  Risk level: {current_risk_level.value}")
        self.logger.debug(f"  Signal strength: {signal_strength}")
        self.logger.debug(f"  Performance multiplier: {performance_multiplier:.2f}")
        
        return position_size

    def _get_performance_multiplier(self) -> Decimal:
        """Calcula un multiplicador basado en la performance reciente."""
        if len(self.daily_pnl_history) < 3:
            return Decimal('1.0')
        
        # Considerar los últimos 5 días
        recent_pnl = self.daily_pnl_history[-5:]
        avg_daily_pnl = sum(recent_pnl) / len(recent_pnl)
        
        # Si la performance es positiva, incrementar ligeramente el riesgo
        if avg_daily_pnl > 0:
            return Decimal('1.1')  # +10%
        # Si la performance es muy negativa, reducir el riesgo
        elif avg_daily_pnl < -self.max_daily_loss / 5:  # Pérdida > 20% del límite diario
            return Decimal('0.8')  # -20%
        else:
            return Decimal('1.0')

    def get_risk_level(self) -> RiskLevel:
        """Determina el nivel de riesgo actual basado en múltiples factores."""
        risk_score = 0
        
        # Factor 1: Pérdidas consecutivas
        if self.consecutive_losses >= 3:
            risk_score += 2
        elif self.consecutive_losses >= 2:
            risk_score += 1
        
        # Factor 2: Drawdown actual
        if self.initial_balance and self.current_balance:
            drawdown_pct = ((self.initial_balance - self.current_balance) / self.initial_balance) * 100
            if drawdown_pct > 15:
                risk_score += 3
            elif drawdown_pct > 10:
                risk_score += 2
            elif drawdown_pct > 5:
                risk_score += 1
        
        # Factor 3: Win rate reciente
        if self.total_trades >= 10:
            recent_win_rate = self.winning_trades / self.total_trades
            if recent_win_rate < 0.3:  # Menos del 30%
                risk_score += 2
            elif recent_win_rate < 0.4:  # Menos del 40%
                risk_score += 1
        
        # Factor 4: Número de trades diarios
        if self.daily_trades > self.max_daily_trades * 0.8:  # Más del 80% del límite
            risk_score += 1
        
        # Determinar nivel de riesgo
        if risk_score >= 6:
            return RiskLevel.EXTREME
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def record_trade(self, trade_result: Dict) -> None:
        """Registra el resultado de un trade."""
        self.total_trades += 1
        self.daily_trades += 1
        
        profit = Decimal(str(trade_result.get('profit', 0)))
        is_win = profit > 0
        
        if is_win:
            self.winning_trades += 1
            self.consecutive_losses = 0
            self.logger.info(f"Trade ganador #{self.total_trades}: +${profit:.2f}")
        else:
            self.consecutive_losses += 1
            self.logger.warning(f"Trade perdedor #{self.total_trades}: ${profit:.2f} (Consecutivas: {self.consecutive_losses})")
        
        # Agregar al historial
        trade_record = {
            'timestamp': datetime.now(),
            'profit': profit,
            'is_win': is_win,
            'consecutive_losses': self.consecutive_losses,
            'balance_after': self.current_balance
        }
        
        self.trade_history.append(trade_record)
        self.last_trade_time = datetime.now()
        
        # Actualizar métricas
        self._update_daily_metrics()

    def _update_daily_metrics(self) -> None:
        """Actualiza las métricas diarias."""
        # Verificar si es un nuevo día
        current_time = datetime.now()
        
        if (self.daily_reset_time is None or 
            current_time.date() > self.daily_reset_time.date()):
            
            # Guardar P&L del día anterior
            if self.daily_start_balance and self.current_balance:
                daily_pnl = self.current_balance - self.daily_start_balance
                self.daily_pnl_history.append(daily_pnl)
                
                # Mantener solo los últimos 30 días
                if len(self.daily_pnl_history) > 30:
                    self.daily_pnl_history = self.daily_pnl_history[-30:]
            
            # Resetear contadores diarios
            self.daily_trades = 0
            self.daily_start_balance = self.current_balance
            self.daily_reset_time = current_time
            
            self.logger.info("Métricas diarias reseteadas")

    def create_stop_loss(self, contract_id: str, stop_type: StopLossType, 
                        params: Dict) -> bool:
        """Crea un stop loss para un contrato específico."""
        try:
            stop_config = {
                'type': stop_type,
                'params': params,
                'created_at': datetime.now(),
                'contract_id': contract_id,
                'triggered': False
            }
            
            # Validar parámetros según el tipo
            if not self._validate_stop_params(stop_type, params):
                self.logger.error(f"Parámetros de stop loss inválidos: {params}")
                return False
            
            self.active_stops[contract_id] = stop_config
            self.logger.info(f"Stop loss {stop_type.value} creado para contrato {contract_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creando stop loss: {e}")
            return False

    def _validate_stop_params(self, stop_type: StopLossType, params: Dict) -> bool:
        """Valida los parámetros de stop loss según su tipo."""
        try:
            if stop_type == StopLossType.FIXED_AMOUNT:
                return 'amount' in params and Decimal(str(params['amount'])) > 0
            
            elif stop_type == StopLossType.PERCENTAGE:
                return 'percentage' in params and 0 < float(params['percentage']) < 1
            
            elif stop_type == StopLossType.TIME_BASED:
                return 'minutes' in params and int(params['minutes']) > 0
            
            elif stop_type == StopLossType.TRAILING:
                return ('distance' in params and 
                        Decimal(str(params['distance'])) > 0)
            
            return True
            
        except (ValueError, KeyError):
            return False

    def check_stop_loss_triggers(self, current_price: Decimal, 
                               contract_id: str) -> Tuple[bool, str]:
        """Verifica si algún stop loss debe activarse."""
        if contract_id not in self.active_stops:
            return False, ""
        
        stop_config = self.active_stops[contract_id]
        
        if stop_config['triggered']:
            return False, "Stop loss ya activado"
        
        stop_type = stop_config['type']
        params = stop_config['params']
        
        # Verificar según el tipo de stop
        if stop_type == StopLossType.TIME_BASED:
            return self._check_time_based_stop(stop_config)
        
        elif stop_type == StopLossType.FIXED_AMOUNT:
            return self._check_fixed_amount_stop(current_price, params)
        
        elif stop_type == StopLossType.PERCENTAGE:
            return self._check_percentage_stop(current_price, params)
        
        return False, ""

    def _check_time_based_stop(self, stop_config: Dict) -> Tuple[bool, str]:
        """Verifica stop loss basado en tiempo."""
        created_at = stop_config['created_at']
        minutes_limit = stop_config['params']['minutes']
        
        elapsed = datetime.now() - created_at
        
        if elapsed.total_seconds() >= (minutes_limit * 60):
            return True, f"Time-based stop: {minutes_limit} minutos transcurridos"
        
        return False, ""

    def _check_fixed_amount_stop(self, current_price: Decimal, 
                               params: Dict) -> Tuple[bool, str]:
        """Verifica stop loss de cantidad fija."""
        # Implementación simplificada
        # En un caso real necesitarías el precio de entrada
        stop_amount = Decimal(str(params['amount']))
        
        # Esta lógica dependería de tener el precio de entrada almacenado
        # Por ahora retornamos False
        return False, ""

    def _check_percentage_stop(self, current_price: Decimal, 
                             params: Dict) -> Tuple[bool, str]:
        """Verifica stop loss porcentual."""
        # Implementación simplificada
        # En un caso real necesitarías el precio de entrada
        stop_percentage = Decimal(str(params['percentage']))
        
        # Esta lógica dependería de tener el precio de entrada almacenado
        # Por ahora retornamos False
        return False, ""

    def remove_stop_loss(self, contract_id: str) -> bool:
        """Elimina un stop loss activo."""
        if contract_id in self.active_stops:
            del self.active_stops[contract_id]
            self.logger.info(f"Stop loss eliminado para contrato {contract_id}")
            return True
        return False

    def get_risk_summary(self) -> Dict:
        """Retorna un resumen completo del estado de riesgo."""
        if not self._validate_balances():
            return {"error": "Balances no configurados"}
        
        daily_loss = self.daily_start_balance - self.current_balance
        daily_loss_pct = (daily_loss / self.daily_start_balance) * 100
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        risk_level = self.get_risk_level()
        
        return {
            'current_balance': self.current_balance,
            'daily_loss': daily_loss,
            'daily_loss_percentage': daily_loss_pct,
            'consecutive_losses': self.consecutive_losses,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'daily_trades': self.daily_trades,
            'risk_level': risk_level.value,
            'can_trade': not self.should_stop_trading()[0],
            'active_stops': len(self.active_stops)
        }