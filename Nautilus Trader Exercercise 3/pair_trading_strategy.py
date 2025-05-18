"""
Pair Trading Strategy Implementation using Nautilus Trader

This module implements a statistical arbitrage strategy that trades pairs of stocks
based on their price relationship. The strategy:
1. Calculates the spread between two stocks
2. Computes z-scores of the spread
3. Enters positions when the z-score exceeds thresholds
4. Exits positions when the spread reverts to mean

Author: Duy Truong
Date: 2025-05-15
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.objects import Quantity
from nautilus_trader.core.data import Data
from nautilus_trader.model import DataType, BarType
from nautilus_trader.model.position import Position
from nautilus_trader.common.enums import LogColor
from nautilus_trader.model.identifiers import InstrumentId

# --- Strategy Configuration ---

class PairTradingConfig(StrategyConfig):
    """
    Configuration for the Pair Trading Strategy.
    
    Attributes:
        instrument_ids: List of two instruments to trade as a pair
        bar_types: List of bar types for each instrument
        enter_threshold: Z-score threshold for entering positions
        exit_threshold: Z-score threshold for exiting positions
        beta: Hedge ratio between the two instruments
        window: Lookback window for spread calculation
        trading_size: Number of shares to trade
    """
    instrument_ids: List[InstrumentId]
    bar_types: List[BarType]
    enter_threshold: float = 1.2
    exit_threshold: float = 0.0
    beta: float = 1.1
    window: int = 10
    trading_size: int = 1

# --- Custom Data Container ---
class PairTradingData(Data):
    """
    Custom data container for pair trading metrics.
    
    Attributes:
        spread: Current spread between the pair
        mean: Rolling mean of the spread
        std: Rolling standard deviation of the spread
        zscore: Current z-score of the spread
    """
    def __init__(
        self,
        spread: float,
        mean: float,
        std: float,
        zscore: float,
        ts_event: int,
        ts_init: int,
    ):
        self.spread = spread
        self.mean = mean
        self.std = std
        self.zscore = zscore
        self._ts_event = ts_event
        self._ts_init = ts_init

# --- Strategy Implementation ---
class PairTradingStrategy(Strategy):
    """
    Statistical arbitrage strategy that trades pairs of stocks based on their price relationship.
    
    The strategy:
    1. Monitors price bars for both instruments
    2. Calculates the spread and its statistics
    3. Enters positions when the spread deviates significantly
    4. Exits positions when the spread reverts to mean
    """
    
    def __init__(self, config: PairTradingConfig):
        """
        Initialize the pair trading strategy.
        
        Args:
            config: Strategy configuration parameters
        """
        super().__init__(config=config)
        self.first_stock_bar_count = 0
        self.second_stock_bar_count = 0
        self.direction = 0  # 0: no position, 1: long-short, -1: short-long

    def on_start(self) -> None:
        """Initialize strategy subscriptions."""
        for bar_type in self.config.bar_types:
            self.subscribe_bars(bar_type)
        self.subscribe_data(DataType(PairTradingData))

    def on_bar(self, bar) -> None:
        """
        Process incoming bar data and calculate pair trading metrics.
        
        Args:
            bar: Price bar data for either instrument
        """
        # Update bar counts and get prices
        if bar.bar_type == self.config.bar_types[0]:
            self.log.info(f"First stock bar: {bar}", LogColor.RED)
            self.first_stock_bar_count += 1
        elif bar.bar_type == self.config.bar_types[1]:
            self.log.info(f"Second stock bar: {bar}", LogColor.RED)
            self.second_stock_bar_count += 1

        # Calculate spread metrics when we have enough data
        if self.first_stock_bar_count == self.second_stock_bar_count:
            self._calculate_spread_metrics()

    def _calculate_spread_metrics(self) -> None:
        """Calculate spread metrics and publish pair trading data."""
        # Get recent price bars
        bars_1 = self.cache.bars(self.config.bar_types[0])[:self.config.window]
        bars_2 = self.cache.bars(self.config.bar_types[1])[:self.config.window]
        
        # Calculate spread
        closes_1 = np.array([float(b.close) for b in bars_1])
        closes_2 = np.array([float(b.close) for b in bars_2])
        spread = closes_2 - self.config.beta * closes_1
        
        # Calculate statistics
        mean = spread.mean()
        std = spread.std()
        zscore = (spread[-1] - mean) / std
        
        # Create and publish pair trading data
        ts_event = max(bars_1[-1].ts_event, bars_2[-1].ts_event)
        ts_init = max(bars_1[-1].ts_init, bars_2[-1].ts_init)
        data = PairTradingData(spread[-1], mean, std, zscore, ts_event, ts_init)
        self.publish_data(DataType(PairTradingData), data)

    def on_data(self, data: PairTradingData) -> None:
        """
        Process pair trading metrics and execute trading logic.
        
        Args:
            data: Pair trading metrics data
        """
        self.log.info(f"Spread mean: {data.mean}", LogColor.YELLOW)
        self.log.info(f"Spread std: {data.std}", LogColor.YELLOW)
        self.log.info(f"Z-score: {data.zscore}", LogColor.YELLOW)
        
        if self.first_stock_bar_count == self.second_stock_bar_count == self.config.window:
            self._try_enter(data)
            self._try_exit(data)

    def _try_enter(self, data: PairTradingData) -> None:
        """
        Attempt to enter pair trading positions based on z-score.
        
        Args:
            data: Pair trading metrics data
        """
        if data.zscore > self.config.enter_threshold and self.direction == 0:
            # Enter long-short position
            self._submit_pair_orders(OrderSide.BUY, OrderSide.SELL)
            self.direction = 1
        elif data.zscore < -self.config.enter_threshold and self.direction == 0:
            # Enter short-long position
            self._submit_pair_orders(OrderSide.SELL, OrderSide.BUY)
            self.direction = -1

    def _try_exit(self, data: PairTradingData) -> None:
        """
        Attempt to exit pair trading positions based on z-score.
        
        Args:
            data: Pair trading metrics data
        """
        if (self.direction == 1 and data.zscore < -self.config.exit_threshold) or \
           (self.direction == -1 and data.zscore > self.config.exit_threshold):
            # Close all positions
            for instrument_id in self.config.instrument_ids:
                self.close_all_positions(instrument_id)
            self.direction = 0

    def _submit_pair_orders(self, side1: OrderSide, side2: OrderSide) -> None:
        """
        Submit orders for both instruments in the pair.
        
        Args:
            side1: Order side for first instrument
            side2: Order side for second instrument
        """
        # Create and submit orders
        order_1 = self.order_factory.market(
            instrument_id=self.config.instrument_ids[0],
            order_side=side1,
            quantity=Quantity.from_int(self.config.trading_size),
        )
        order_2 = self.order_factory.market(
            instrument_id=self.config.instrument_ids[1],
            order_side=side2,
            quantity=Quantity.from_int(self.config.trading_size),
        )
        self.submit_order(order_1)
        self.submit_order(order_2)

    def on_stop(self) -> None:
        """Clean up strategy subscriptions and positions."""
        self.unsubscribe_data(DataType(PairTradingData))
        for bar_type in self.config.bar_types:
            self.unsubscribe_bars(bar_type)
        for instrument_id in self.config.instrument_ids:
            self.close_all_positions(instrument_id) 
