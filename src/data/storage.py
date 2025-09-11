"""
Sistema de almacenamiento de datos para el trading bot.
Maneja persistencia de datos históricos, configuraciones y métricas.
"""

import os
import json
import csv
import sqlite3
import logging
import pickle
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import threading


class StorageManager:
    """Gestor principal de almacenamiento de datos."""
    
    def __init__(self, data_dir: str = "data"):
        """Inicializa el gestor de almacenamiento."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "trading_bot.db")
        
        # Thread lock para operaciones de BD
        self._db_lock = threading.Lock()
        
        # Crear directorio si no existe
        os.makedirs(data_dir, exist_ok=True)
        
        # Inicializar base de datos
        self._init_database()
        
        self.logger.info(f"StorageManager inicializado - Data dir: {data_dir}")

    def _init_database(self) -> None:
        """Inicializa las tablas de la base de datos."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Tabla de trades
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        contract_type VARCHAR(10) NOT NULL,
                        stake DECIMAL(10,2) NOT NULL,
                        payout DECIMAL(10,2),
                        profit DECIMAL(10,2) NOT NULL,
                        is_win BOOLEAN NOT NULL,
                        contract_id VARCHAR(50),
                        entry_price DECIMAL(15,5),
                        exit_price DECIMAL(15,5),
                        duration INTEGER,
                        signal_reason TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabla de datos de mercado
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        open_price DECIMAL(15,5) NOT NULL,
                        high_price DECIMAL(15,5) NOT NULL,
                        low_price DECIMAL(15,5) NOT NULL,
                        close_price DECIMAL(15,5) NOT NULL,
                        granularity INTEGER NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(timestamp, symbol, granularity)
                    )
                ''')
                
                # Tabla de balance histórico
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS balance_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        balance DECIMAL(15,2) NOT NULL,
                        daily_pnl DECIMAL(15,2),
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabla de configuraciones
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS configurations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name VARCHAR(50) NOT NULL UNIQUE,
                        config_data TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabla de eventos del sistema
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        event_type VARCHAR(50) NOT NULL,
                        description TEXT,
                        data TEXT,
                        severity VARCHAR(20) DEFAULT 'INFO',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                
                # Crear índices para mejor rendimiento
                self._create_indexes(cursor)
                
                self.logger.info("Base de datos inicializada correctamente")
                
        except Exception as e:
            self.logger.error(f"Error inicializando base de datos: {e}")
            raise

    def _create_indexes(self, cursor) -> None:
        """Crea índices para optimizar consultas."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_balance_timestamp ON balance_history(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                self.logger.warning(f"Error creando índice: {e}")

    @contextmanager
    def _get_db_connection(self):
        """Context manager para conexiones de base de datos thread-safe."""
        with self._db_lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row  # Para acceso por nombre de columna
                yield conn
            except Exception as e:
                if conn:
                    conn.rollback()
                raise e
            finally:
                if conn:
                    conn.close()

    def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Guarda información de un trade en la base de datos."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trades (
                        timestamp, symbol, contract_type, stake, payout, profit,
                        is_win, contract_id, entry_price, exit_price, duration, signal_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data.get('timestamp', datetime.now()),
                    trade_data.get('symbol', ''),
                    trade_data.get('contract_type', ''),
                    float(trade_data.get('stake', 0)),
                    float(trade_data.get('payout', 0)) if trade_data.get('payout') else None,
                    float(trade_data.get('profit', 0)),
                    bool(trade_data.get('is_win', False)),
                    trade_data.get('contract_id', ''),
                    float(trade_data.get('entry_price', 0)) if trade_data.get('entry_price') else None,
                    float(trade_data.get('exit_price', 0)) if trade_data.get('exit_price') else None,
                    trade_data.get('duration', 0),
                    trade_data.get('signal_reason', '')
                ))
                
                conn.commit()
                self.logger.debug(f"Trade guardado: {trade_data.get('contract_id', 'N/A')}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error guardando trade: {e}")
            return False

    def save_market_data(self, candle_data: Dict[str, Any], symbol: str, granularity: int) -> bool:
        """Guarda datos de mercado (velas) en la base de datos."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data (
                        timestamp, symbol, open_price, high_price, low_price, close_price, granularity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    candle_data.get('timestamp', datetime.now()),
                    symbol,
                    float(candle_data.get('open', 0)),
                    float(candle_data.get('high', 0)),
                    float(candle_data.get('low', 0)),
                    float(candle_data.get('close', 0)),
                    granularity
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error guardando datos de mercado: {e}")
            return False

    def save_balance_snapshot(self, balance: Decimal, daily_pnl: Optional[Decimal] = None,
                            total_trades: int = 0, winning_trades: int = 0) -> bool:
        """Guarda un snapshot del balance actual."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO balance_history (timestamp, balance, daily_pnl, total_trades, winning_trades)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now(),
                    float(balance),
                    float(daily_pnl) if daily_pnl else None,
                    total_trades,
                    winning_trades
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error guardando balance: {e}")
            return False

    def get_trades_history(self, days: int = 30, symbol: Optional[str] = None) -> List[Dict]:
        """Obtiene el historial de trades."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                where_clause = "WHERE timestamp >= ?"
                params = [datetime.now() - timedelta(days=days)]
                
                if symbol:
                    where_clause += " AND symbol = ?"
                    params.append(symbol)
                
                cursor.execute(f'''
                    SELECT * FROM trades 
                    {where_clause}
                    ORDER BY timestamp DESC
                ''', params)
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error obteniendo historial de trades: {e}")
            return []

    def get_market_data(self, symbol: str, granularity: int, 
                       hours: int = 24) -> List[Dict]:
        """Obtiene datos históricos de mercado."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND granularity = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                ''', (
                    symbol,
                    granularity,
                    datetime.now() - timedelta(hours=hours)
                ))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de mercado: {e}")
            return []

    def get_trading_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Calcula estadísticas de trading."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                start_date = datetime.now() - timedelta(days=days)
                
                # Estadísticas básicas
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN is_win THEN 1 ELSE 0 END) as winning_trades,
                        SUM(profit) as total_profit,
                        AVG(profit) as avg_profit,
                        MAX(profit) as max_profit,
                        MIN(profit) as min_profit
                    FROM trades 
                    WHERE timestamp >= ?
                ''', (start_date,))
                
                stats = dict(cursor.fetchone())
                
                # Win rate
                stats['win_rate'] = (stats['winning_trades'] / stats['total_trades'] * 100 
                                   if stats['total_trades'] > 0 else 0)
                
                # Profit factor
                cursor.execute('''
                    SELECT 
                        SUM(CASE WHEN profit > 0 THEN profit ELSE 0 END) as gross_profit,
                        SUM(CASE WHEN profit < 0 THEN ABS(profit) ELSE 0 END) as gross_loss
                    FROM trades 
                    WHERE timestamp >= ?
                ''', (start_date,))
                
                pf_data = dict(cursor.fetchone())
                stats['profit_factor'] = (pf_data['gross_profit'] / pf_data['gross_loss'] 
                                        if pf_data['gross_loss'] > 0 else 0)
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error calculando estadísticas: {e}")
            return {}

    def log_system_event(self, event_type: str, description: str, 
                        data: Optional[Dict] = None, severity: str = "INFO") -> bool:
        """Registra un evento del sistema."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_events (timestamp, event_type, description, data, severity)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now(),
                    event_type,
                    description,
                    json.dumps(data, default=str) if data else None,
                    severity
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error registrando evento: {e}")
            return False

    def save_configuration(self, name: str, config: Dict[str, Any]) -> bool:
        """Guarda una configuración con nombre."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO configurations (name, config_data, updated_at)
                    VALUES (?, ?, ?)
                ''', (
                    name,
                    json.dumps(config, default=str),
                    datetime.now()
                ))
                
                conn.commit()
                self.logger.info(f"Configuración '{name}' guardada")
                return True
                
        except Exception as e:
            self.logger.error(f"Error guardando configuración: {e}")
            return False

    def load_configuration(self, name: str) -> Optional[Dict[str, Any]]:
        """Carga una configuración por nombre."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT config_data FROM configurations WHERE name = ?
                ''', (name,))
                
                row = cursor.fetchone()
                if row:
                    return json.loads(row['config_data'])
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error cargando configuración: {e}")
            return None

    def export_trades_to_csv(self, filepath: str, days: int = 30) -> bool:
        """Exporta trades a un archivo CSV."""
        try:
            trades = self.get_trades_history(days)
            
            if not trades:
                self.logger.warning("No hay trades para exportar")
                return False
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = trades[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for trade in trades:
                    writer.writerow(trade)
            
            self.logger.info(f"Trades exportados a: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exportando trades: {e}")
            return False

    def cleanup_old_data(self, days_to_keep: int = 90) -> bool:
        """Elimina datos antiguos para mantener la base de datos limpia."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Limpiar datos antiguos
                tables_to_clean = [
                    'market_data',
                    'system_events'
                ]
                
                for table in tables_to_clean:
                    cursor.execute(f'''
                        DELETE FROM {table} WHERE timestamp < ?
                    ''', (cutoff_date,))
                
                # Limpiar balance history muy antiguo (mantener solo snapshots semanales)
                cursor.execute('''
                    DELETE FROM balance_history 
                    WHERE timestamp < ? 
                    AND id NOT IN (
                        SELECT id FROM balance_history 
                        WHERE timestamp < ?
                        AND strftime('%w', timestamp) = '0'  -- Solo domingos
                        ORDER BY timestamp DESC
                    )
                ''', (cutoff_date, cutoff_date))
                
                conn.commit()
                
                changes = cursor.rowcount
                self.logger.info(f"Limpieza completada: {changes} registros eliminados")
                return True
                
        except Exception as e:
            self.logger.error(f"Error en limpieza de datos: {e}")
            return False

    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """Crea un backup de la base de datos."""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = os.path.join(self.data_dir, f"backup_{timestamp}.db")
            
            # Crear backup usando SQLite
            with self._get_db_connection() as conn:
                with sqlite3.connect(backup_path) as backup_conn:
                    conn.backup(backup_conn)
            
            self.logger.info(f"Backup creado: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creando backup: {e}")
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """Obtiene información sobre la base de datos."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Contar registros en cada tabla
                tables = ['trades', 'market_data', 'balance_history', 'configurations', 'system_events']
                table_counts = {}
                
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    table_counts[table] = cursor.fetchone()[0]
                
                # Tamaño del archivo de base de datos
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    'database_path': self.db_path,
                    'database_size_mb': round(db_size / (1024 * 1024), 2),
                    'table_counts': table_counts,
                    'total_records': sum(table_counts.values())
                }
                
        except Exception as e:
            self.logger.error(f"Error obteniendo info de BD: {e}")
            return {}


class FileStorageManager:
    """Maneja almacenamiento de archivos (configuraciones, logs, etc.)."""
    
    def __init__(self, base_dir: str = "data"):
        """Inicializa el gestor de archivos."""
        self.base_dir = base_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Crear subdirectorios
        self.subdirs = {
            'configs': os.path.join(base_dir, 'configs'),
            'exports': os.path.join(base_dir, 'exports'),
            'cache': os.path.join(base_dir, 'cache'),
            'backups': os.path.join(base_dir, 'backups')
        }
        
        for subdir in self.subdirs.values():
            os.makedirs(subdir, exist_ok=True)

    def save_json(self, data: Dict, filename: str, subdir: str = 'configs') -> bool:
        """Guarda datos en formato JSON."""
        try:
            filepath = os.path.join(self.subdirs.get(subdir, self.base_dir), filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.debug(f"JSON guardado: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando JSON: {e}")
            return False

    def load_json(self, filename: str, subdir: str = 'configs') -> Optional[Dict]:
        """Carga datos desde un archivo JSON."""
        try:
            filepath = os.path.join(self.subdirs.get(subdir, self.base_dir), filename)
            
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error cargando JSON: {e}")
            return None

    def save_pickle(self, data: Any, filename: str, subdir: str = 'cache') -> bool:
        """Guarda datos usando pickle (para objetos complejos)."""
        try:
            filepath = os.path.join(self.subdirs.get(subdir, self.base_dir), filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando pickle: {e}")
            return False

    def load_pickle(self, filename: str, subdir: str = 'cache') -> Optional[Any]:
        """Carga datos desde un archivo pickle."""
        try:
            filepath = os.path.join(self.subdirs.get(subdir, self.base_dir), filename)
            
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error cargando pickle: {e}")
            return None

    def cleanup_old_files(self, subdir: str, days_old: int = 30) -> int:
        """Limpia archivos antiguos en un subdirectorio."""
        try:
            directory = self.subdirs.get(subdir, self.base_dir)
            cutoff_time = datetime.now() - timedelta(days=days_old)
            
            cleaned_count = 0
            
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        cleaned_count += 1
                        self.logger.debug(f"Archivo eliminado: {filename}")
            
            if cleaned_count > 0:
                self.logger.info(f"Limpieza de {subdir}: {cleaned_count} archivos eliminados")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error limpiando archivos: {e}")
            return 0