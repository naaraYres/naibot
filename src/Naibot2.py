import time
import logging
from decimal import Decimal
from typing import Dict, Optional

from .exchanges.deriv_client import DerivWebSocketClient
from .strategies.confluence_strategy import ConfluenceStrategy
from .risk_management.position_sizing import RiskManager
from .utils.bot_states import BotState

class TradingBot:
    """Bot de trading optimizado con mejor gestión de estados y errores."""
    
    def __init__(self, config: Dict):
        """Inicializa el bot con validación exhaustiva."""
        self.config = config
        self.logger = logging.getLogger("TradingBot")
        
        # Validar configuración
        self.validate_config()
        
        # Componentes principales
        self.ws_client = DerivWebSocketClient(
            app_id=config['app_id'],
            token=config['token'],
            symbol=config['symbol'],
            granularity=int(config['granularity'])
        )
        
        self.strategy = ConfluenceStrategy(
            symbol=config['symbol'],
            strategy_params=config['strategy_params'][config['symbol']],
            tolerancia_zona=Decimal(str(config['tolerancia_zona'])),
            ema_fast_period=int(config['ema_fast_period']),
            ema_slow_period=int(config['ema_slow_period'])
        )
        
        # Parámetros de trading
        self.stake = Decimal(str(config['stake']))
        self.duration = int(config['duration'])
        self.duration_unit = config['duration_unit']
        
        # Gestión de riesgo
        self.risk_manager = RiskManager(
            max_daily_loss=Decimal(str(config['max_daily_loss'])),
            max_consecutive_losses=int(config['max_consecutive_losses'])
        )
        
        # Estado del bot
        self.bot_state = BotState.BUSCANDO_SENAL
        self.equity_start = None
        self.current_balance = None
        self.total_trades = 0
        self.winning_trades = 0
        self.current_contract_id = None
        
        # Timeouts y reintentos
        self.proposal_timeout = 30  # segundos
        self.buy_timeout = 20  # segundos
        
        self.logger.info("Bot de trading inicializado correctamente")

    def validate_config(self) -> None:
        """Validación exhaustiva de configuración."""
        required_keys = [
            'app_id', 'token', 'symbol', 'stake', 'duration', 'duration_unit',
            'granularity', 'max_daily_loss', 'max_consecutive_losses',
            'tolerancia_zona', 'ema_fast_period', 'ema_slow_period', 'strategy_params'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Falta el parámetro de configuración: {key}")
        
        # Validar parámetros específicos
        if self.config['symbol'] not in self.config['strategy_params']:
            raise ValueError(f"No hay parámetros de estrategia para {self.config['symbol']}")
        
        try:
            stake = Decimal(str(self.config['stake']))
            if stake <= 0:
                raise ValueError("El stake debe ser mayor a 0")
        except:
            raise ValueError("El stake debe ser un número válido")
        
        try:
            granularity = int(self.config['granularity'])
            if granularity not in [60, 120, 180, 300, 600, 900, 1800, 3600, 14400, 86400]:
                raise ValueError("Granularidad no válida para Deriv")
        except:
            raise ValueError("La granularidad debe ser un número entero válido")

    def place_option(self, contract_type: str, analysis_info: Dict) -> bool:
        """Coloca una opción con validación mejorada."""
        try:
            proposal = {
                "proposal": 1,
                "amount": str(self.stake),
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": self.duration,
                "duration_unit": self.duration_unit,
                "symbol": self.config['symbol']
            }
            
            if self.ws_client.send(proposal):
                self.logger.info(f"📤 Solicitando propuesta {contract_type} por ${self.stake}")
                self.logger.info(f"📊 Zona: {analysis_info.get('signal_zone', 'N/A')} | "
                              f"Fuerza: {analysis_info.get('signal_strength', 0)}")
                return True
            else:
                self.logger.error("Error al enviar propuesta")
                return False
                
        except Exception as e:
            self.logger.error(f"Error al crear propuesta: {e}")
            return False

    def process_api_response(self, data: Dict) -> Dict:
        """Procesa respuestas de la API con mejor manejo de errores."""
        if data.get('error'):
            error_msg = data['error'].get('message', 'Error desconocido')
            error_code = data['error'].get('code', 'N/A')
            self.logger.error(f"❌ Error API [{error_code}]: {error_msg}")
            return {"status": "error", "message": error_msg}

        msg_type = data.get('msg_type')

        if msg_type == 'proposal':
            return self._handle_proposal_response(data)
        elif msg_type == 'buy':
            return self._handle_buy_response(data)
        elif msg_type == 'proposal_open_contract':
            return self._handle_contract_update(data)
        else:
            return {"status": "irrelevant"}

    def _handle_proposal_response(self, data: Dict) -> Dict:
        """Maneja respuesta de propuesta."""
        try:
            proposal_data = data.get('proposal', {})
            proposal_id = proposal_data.get('id')
            ask_price = proposal_data.get('ask_price')
            
            if not proposal_id or ask_price is None:
                self.logger.error("Respuesta de propuesta inválida")
                return {"status": "error", "message": "Propuesta inválida"}
            
            price = Decimal(str(ask_price))
            payout = proposal_data.get('payout', 0)
            
            self.logger.info(f"💰 Propuesta recibida - Precio: ${price} | Payout: ${payout}")
            
            # Verificar si el precio es razonable (no más del 10% del payout)
            if payout > 0 and price > (Decimal(str(payout)) * Decimal('0.1')):
                self.logger.warning(f"⚠️ Precio alto detectado: ${price} vs payout ${payout}")
            
            buy_order = {
                "buy": proposal_id,
                "price": str(price)
            }
            
            if self.ws_client.send(buy_order):
                self.logger.info(f"📤 Enviando orden de compra...")
                return {"status": "buying", "price": price}
            else:
                return {"status": "error", "message": "Error al enviar orden"}
                
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error procesando propuesta: {e}")
            return {"status": "error", "message": f"Error en propuesta: {e}"}

    def _handle_buy_response(self, data: Dict) -> Dict:
        """Maneja respuesta de compra."""
        try:
            buy_data = data.get('buy', {})
            contract_id = buy_data.get('contract_id')
            buy_price = buy_data.get('buy_price')
            
            if not contract_id:
                self.logger.error("ID de contrato no recibido")
                return {"status": "error", "message": "Sin ID de contrato"}
            
            self.current_contract_id = contract_id
            self.total_trades += 1
            
            self.logger.info(f"✅ Operación ejecutada - ID: {contract_id}")
            self.logger.info(f"💳 Precio de compra: ${buy_price}")
            self.logger.info(f"📈 Total de operaciones: {self.total_trades}")
            
            # Suscribirse a actualizaciones del contrato
            subscribe_msg = {
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1
            }
            
            if self.ws_client.send(subscribe_msg):
                return {"status": "bought", "contract_id": contract_id}
            else:
                self.logger.error("Error al suscribirse al contrato")
                return {"status": "error", "message": "Error en suscripción"}
                
        except (KeyError, ValueError) as e:
            self.logger.error(f"Error procesando compra: {e}")
            return {"status": "error", "message": f"Error en compra: {e}"}

    def _handle_contract_update(self, data: Dict) -> Dict:
        """Maneja actualizaciones del contrato."""
        try:
            contract_data = data.get('proposal_open_contract', {})
            is_sold = contract_data.get('is_sold', False)
            
            if not is_sold:
                # Actualización intermedia del contrato
                current_spot = contract_data.get('current_spot')
                if current_spot:
                    self.logger.debug(f"📊 Spot actual: {current_spot}")
                return {"status": "monitoring"}
            
            # Contrato cerrado
            profit = contract_data.get('profit', 0)
            balance_after = contract_data.get('balance_after')
            exit_tick = contract_data.get('exit_tick')
            
            profit_decimal = Decimal(str(profit))
            is_win = profit_decimal > 0
            
            # Actualizar estadísticas
            if is_win:
                self.winning_trades += 1
                win_rate = (self.winning_trades / self.total_trades) * 100
                self.logger.info(f"🎉 ¡GANAMOS! Ganancia: ${profit_decimal:.2f}")
                self.logger.info(f"📊 Tasa de éxito: {win_rate:.1f}% ({self.winning_trades}/{self.total_trades})")
            else:
                loss_rate = ((self.total_trades - self.winning_trades) / self.total_trades) * 100
                self.logger.error(f"❌ PERDIMOS. Pérdida: ${profit_decimal:.2f}")
                self.logger.error(f"📊 Tasa de pérdida: {loss_rate:.1f}%")
            
            # Actualizar gestión de riesgo
            self.risk_manager.update_consecutive_losses(is_win)
            
            # Actualizar balance
            if balance_after:
                self.current_balance = Decimal(str(balance_after))
                self.logger.info(f"💰 Balance actual: ${self.current_balance:.2f}")
            
            # Verificar límites de riesgo
            if self.equity_start and self.current_balance:
                daily_loss = self.equity_start - self.current_balance
                consecutive_losses = self.risk_manager.get_consecutive_losses()
                
                should_stop, reason = self.risk_manager.should_stop_trading(daily_loss, consecutive_losses)
                if should_stop:
                    return {"status": "stop", "reason": reason}
            
            # Información adicional
            if exit_tick:
                self.logger.info(f"🎯 Precio de salida: {exit_tick}")
            
            self.logger.error(f"📊 Pérdidas consecutivas: {self.risk_manager.get_consecutive_losses()}")
            
            return {
                "status": "closed", 
                "is_win": is_win, 
                "profit": profit_decimal,
                "balance": self.current_balance
            }
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error procesando actualización de contrato: {e}")
            return {"status": "error", "message": f"Error en contrato: {e}"}

    def _log_bot_status(self) -> None:
        """Registra el estado actual del bot."""
        if self.equity_start and self.current_balance:
            daily_pnl = self.current_balance - self.equity_start
            daily_pnl_pct = (daily_pnl / self.equity_start) * 100
            
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            self.logger.info(f"📊 RESUMEN DEL BOT:")
            self.logger.info(f"   Estado: {self.bot_state.value}")
            self.logger.info(f"   Balance inicial: ${self.equity_start:.2f}")
            self.logger.info(f"   Balance actual: ${self.current_balance:.2f}")
            self.logger.info(f"   P&L del día: ${daily_pnl:.2f} ({daily_pnl_pct:+.2f}%)")
            self.logger.info(f"   Operaciones: {self.total_trades} | Ganadas: {self.winning_trades}")
            self.logger.info(f"   Tasa de éxito: {win_rate:.1f}%")
            self.logger.info(f"   Pérdidas consecutivas: {self.risk_manager.get_consecutive_losses()}")

    def run(self) -> None:
        """Ejecuta el bucle principal del bot con gestión robusta de errores."""
        try:
            self.logger.info("🚀 Iniciando bot de trading...")
            
            # Establecer conexión
            if not self._initialize_connection():
                return
            
            # Configurar suscripciones
            if not self.ws_client.subscribe_candles():
                self.logger.error("Error al suscribirse a datos de mercado")
                return
            
            self._log_startup_info()
            
            # Bucle principal
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            while self.bot_state != BotState.DETENIDO:
                try:
                    # Recibir datos con timeout
                    data = self.ws_client.receive(timeout=30.0)
                    
                    if not data:
                        self.logger.warning("⏰ Timeout o conexión perdida")
                        if not self._handle_connection_loss():
                            break
                        continue
                    
                    # Resetear contador de errores en recepción exitosa
                    consecutive_errors = 0
                    
                    # Procesar respuesta de la API
                    response = self.process_api_response(data)
                    
                    if not self._handle_bot_response(response):
                        break
                    
                    # Procesar datos de mercado
                    if self.bot_state == BotState.BUSCANDO_SENAL:
                        self._process_market_data(data)
                    
                    # Log periódico del estado
                    if self.total_trades > 0 and self.total_trades % 10 == 0:
                        self._log_bot_status()
                
                except KeyboardInterrupt:
                    self.logger.info("🛑 Bot detenido por el usuario")
                    break
                    
                except Exception as e:
                    consecutive_errors += 1
                    self.logger.error(f"❌ Error en bucle principal ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("🛑 Demasiados errores consecutivos. Deteniendo bot.")
                        break
                    
                    # Esperar antes de continuar
                    time.sleep(1.0)
                
                # Pequeña pausa para evitar sobrecarga del CPU
                time.sleep(0.05)
        
        finally:
            self._cleanup()

    def _initialize_connection(self) -> bool:
        """Inicializa la conexión y obtiene información de la cuenta."""
        if not self.ws_client.connect():
            self.logger.error("❌ No se pudo conectar con Deriv")
            return False
        
        # Obtener respuesta de autorización
        auth_response = self.ws_client.receive(timeout=10.0)
        if not auth_response:
            self.logger.error("❌ No se recibió respuesta de autorización")
            return False
        
        if auth_response.get('error'):
            error_msg = auth_response['error'].get('message', 'Error desconocido')
            self.logger.error(f"❌ Error de autorización: {error_msg}")
            return False
        
        # Extraer información de la cuenta
        auth_data = auth_response.get('authorize', {})
        self.equity_start = Decimal(str(auth_data.get('balance', 0)))
        self.current_balance = self.equity_start
        
        account_info = {
            'loginid': auth_data.get('loginid', 'N/A'),
            'currency': auth_data.get('currency', 'USD'),
            'country': auth_data.get('country', 'N/A'),
            'email': auth_data.get('email', 'N/A')
        }
        
        self.logger.info(f"✅ Conectado exitosamente")
        self.logger.info(f"👤 Cuenta: {account_info['loginid']} ({account_info['country']})")
        self.logger.info(f"💰 Balance inicial: ${self.equity_start:.2f} {account_info['currency']}")
        
        return True

    def _log_startup_info(self) -> None:
        """Registra información de configuración al inicio."""
        self.logger.info("⚙️ CONFIGURACIÓN DEL BOT:")
        self.logger.info(f"   Símbolo: {self.config['symbol']}")
        self.logger.info(f"   Stake por operación: ${self.stake}")
        self.logger.info(f"   Duración: {self.duration} {self.duration_unit}")
        self.logger.info(f"   Granularidad: {self.config['granularity']}s")
        self.logger.info(f"   Límite pérdida diaria: ${self.risk_manager.max_daily_loss}")
        self.logger.info(f"   Límite pérdidas consecutivas: {self.risk_manager.max_consecutive_losses}")
        self.logger.info(f"   Tolerancia de zona: {self.strategy.tolerancia_zona}")
        self.logger.info(f"   EMAs: {self.strategy.ema_fast_period}/{self.strategy.ema_slow_period}")
        self.logger.info("🎯 Esperando datos de mercado...")

    def _handle_connection_loss(self) -> bool:
        """Maneja la pérdida de conexión."""
        self.logger.warning("🔄 Intentando reconectar...")
        self.bot_state = BotState.ERROR
        
        if self.ws_client.reconnect():
            self.bot_state = BotState.BUSCANDO_SENAL
            self.logger.info("✅ Reconexión exitosa")
            return True
        else:
            self.logger.error("❌ No se pudo reconectar. Deteniendo bot.")
            self.bot_state = BotState.DETENIDO
            return False

    def _handle_bot_response(self, response: Dict) -> bool:
        """Maneja las respuestas del procesamiento de API."""
        status = response.get("status")
        
        if status == "buying":
            self.bot_state = BotState.ESPERANDO_COMPRA
            
        elif status == "bought":
            self.bot_state = BotState.OPERACION_ABIERTA
            
        elif status == "closed":
            self.bot_state = BotState.BUSCANDO_SENAL
            
        elif status == "stop":
            reason = response.get("reason", "unknown")
            self.logger.error(f"🛑 Bot detenido por: {reason}")
            self.bot_state = BotState.DETENIDO
            return False
            
        elif status == "error":
            error_msg = response.get("message", "Error desconocido")
            self.logger.error(f"⚠️ Error procesando respuesta: {error_msg}")
            self.bot_state = BotState.BUSCANDO_SENAL  # Continuar buscando señales
        
        return True

    def _process_market_data(self, data: Dict) -> None:
        """Procesa datos de mercado y busca señales."""
        if data.get('msg_type') != 'ohlc':
            return
        
        ohlc = data.get('ohlc')
        if not ohlc:
            return
        
        try:
            # Verificar que no estemos en una operación
            if self.bot_state != BotState.BUSCANDO_SENAL:
                return
            
            timestamp = int(ohlc.get('open_time', 0))
            
            # Evitar operar en la misma vela
            if not self.strategy.should_trade(timestamp):
                return
            
            # Crear estructura de vela
            new_candle = {
                'open': float(ohlc.get('open', 0)),
                'high': float(ohlc.get('high', 0)),
                'low': float(ohlc.get('low', 0)),
                'close': float(ohlc.get('close', 0))
            }
            
            # Analizar mercado
            signal_result = self.strategy.analyze(new_candle)
            
            if signal_result:
                signal_type, reason, analysis_info = signal_result
                
                self.logger.info(f"🎯 SEÑAL DETECTADA: {signal_type}")
                self.logger.info(f"📝 Razón: {reason}")
                
                # Colocar la orden
                if self.place_option(signal_type, analysis_info):
                    self.bot_state = BotState.ESPERANDO_COMPRA
                else:
                    self.logger.error("❌ Error al colocar la orden")
        
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error procesando datos de mercado: {e}")

    def _cleanup(self) -> None:
        """Limpia recursos al finalizar."""
        self.logger.info("🧹 Limpiando recursos...")
        
        try:
            if self.ws_client:
                self.ws_client.close()
        except Exception as e:
            self.logger.error(f"Error al cerrar conexión: {e}")
        
        # Log final
        self._log_bot_status()
        self.logger.info("👋 Bot de trading finalizado")