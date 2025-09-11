import logging
from decimal import Decimal
from typing import Tuple

class RiskManager:
    """Gestión de riesgo del bot."""
    
    def __init__(self, max_daily_loss: Decimal, max_consecutive_losses: int):
        """Inicializa el gestor de riesgo."""
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses
        self.consecutive_losses = 0
        self.logger = logging.getLogger("RiskManager")
    
    def should_stop_trading(self, daily_loss: Decimal, consecutive_losses: int) -> Tuple[bool, str]:
        """Determina si detener trading por límites de riesgo."""
        # Verificar límite de pérdida diaria
        if daily_loss >= self.max_daily_loss:
            self.logger.error(f"🛑 Límite de pérdida diaria alcanzado: ${daily_loss:.2f}")
            return True, "daily_loss_limit"
        
        # Verificar límite de pérdidas consecutivas
        if consecutive_losses >= self.max_consecutive_losses:
            self.logger.error(f"🛑 Límite de pérdidas consecutivas: {consecutive_losses}")
            return True, "consecutive_losses"
        
        return False, ""
    
    def update_consecutive_losses(self, is_win: bool) -> None:
        """Actualiza el contador de pérdidas consecutivas."""
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
    
    def get_consecutive_losses(self) -> int:
        """Obtiene el número actual de pérdidas consecutivas."""
        return self.consecutive_losses