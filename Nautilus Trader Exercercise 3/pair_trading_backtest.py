"""
Backtesting Setup for Pair Trading Strategy

This module provides the configuration and setup for backtesting the pair trading strategy
using Nautilus Trader's backtesting engine.

Author: [Your Name]
Date: [Current Date]
"""

from pathlib import Path
import databento as db

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.enums import OmsType, AccountType
from nautilus_trader.model import BarType
from decimal import Decimal

from pair_trading_strategy import PairTradingConfig
from my_config import DATABENTO_API_KEY


def setup_backtest_engine(
    instrument_ids: list[InstrumentId],
    bar_types: list[BarType],
    databento_api_key: str,
    data_path: Path = Path("data/databento"),
) -> BacktestEngine:
    """
    Set up and configure the backtesting engine for pair trading.
    
    Args:
        instrument_ids: List of instruments to trade
        bar_types: List of bar types for each instrument
        databento_api_key: Databento API key for data access
        data_path: Path to store downloaded data
        
    Returns:
        Configured backtesting engine
    """
    # Create data directory
    data_path.mkdir(exist_ok=True)
    
    # Configure strategy
    strategy_config = ImportableStrategyConfig(
        strategy_path="pair_trading_strategy:PairTradingStrategy",
        config_path="pair_trading_strategy:PairTradingConfig",
        config={
            "instrument_ids": instrument_ids,
            "bar_types": bar_types,
            "enter_threshold": 1.2,
            "exit_threshold": 0.0,
            "beta": 1.1,
            "window": 10,
            "trading_size": 1,
        }
    )
    
    # Configure engine
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTEST-PAIRTRADING-001"),
        strategies=[strategy_config],
        logging=LoggingConfig(log_level="DEBUG")
    )
    
    # Create engine
    engine = BacktestEngine(config=engine_config)
    
    # Add venue
    engine.add_venue(
        venue=Venue("XNAS"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, USD)],
        base_currency=USD,
        default_leverage=Decimal(1)
    )
    
    # Add instruments
    for instrument_id in instrument_ids:
        engine.add_instrument(
            Equity(
                instrument_id=instrument_id,
                raw_symbol=instrument_id.symbol,
                currency=USD,
                price_precision=2,
                price_increment=Price.from_str("0.01"),
                lot_size=Quantity.from_int(1),
                ts_event=0,
                ts_init=0
            )
        )
    
    # Download and load data
    load_historical_data(
        engine=engine,
        instrument_ids=instrument_ids,
        api_key=databento_api_key,
        data_path=data_path
    )
    
    return engine

def load_historical_data(
    engine: BacktestEngine,
    instrument_ids: list[InstrumentId],
    api_key: str,
    data_path: Path,
    start_time: str = "2024-01-09T09:30-04:00",
    end_time: str = "2025-01-09T10:00-04:00",
) -> None:
    """
    Download and load historical data for backtesting.
    
    Args:
        engine: Backtesting engine
        instrument_ids: List of instruments to load data for
        api_key: Databento API key
        data_path: Path to store downloaded data
        start_time: Start time for data download
        end_time: End time for data download
    """
    # Initialize Databento client
    client = db.Historical(api_key)
    
    # Prepare data path
    path = data_path / "hist.trades.dbn.zst"
    
    # Download data if not exists
    if not path.exists():
        symbols = [str(id.symbol) for id in instrument_ids]
        client.timeseries.get_range(
            dataset="XNAS.ITCH",
            symbols=symbols,
            schema="ohlcv-1d",
            start=start_time,
            end=end_time,
            path=path,
        )
    
    # Load data into engine
    from nautilus_trader.adapters.databento.loaders import DatabentoDataLoader
    loader = DatabentoDataLoader()
    
    data = []
    for instrument_id in instrument_ids:
        data.extend(loader.from_dbn_file(
            path=path,
            instrument_id=instrument_id,
            as_legacy_cython=True,
        ))
    
    engine.add_data(data)

def main():
    """Run the pair trading backtest."""
    # Define instruments and bar types
    instrument_ids = [
        InstrumentId.from_str("PEP.XNAS"),
        InstrumentId.from_str("KO.XNAS")
    ]
    bar_types = [
        BarType.from_str("PEP.XNAS-1-DAY-LAST-EXTERNAL"),
        BarType.from_str("KO.XNAS-1-DAY-LAST-EXTERNAL")
    ]
    
    
    # Set up and run backtest
    engine = setup_backtest_engine(
        instrument_ids=instrument_ids,
        bar_types=bar_types,
        databento_api_key=DATABENTO_API_KEY
    )
    
    # Run backtest
    engine.run()
    

if __name__ == "__main__":
    main() 