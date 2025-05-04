import numpy as np
from decimal import Decimal

from nautilus_trader.core.data import Data
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig, BacktestEngineConfig, LoggingConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model import InstrumentId, BarType, Bar, DataType, TraderId
from nautilus_trader.model.enums import OrderSide, PositionSide, OmsType, AccountType
from nautilus_trader.model.identifiers import Symbol, Venue
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.config import ImportableStrategyConfig

# --------------------
# 1. Custom Data Class
# --------------------
class ZFromNBars(Data):
    """
    Holds mean, std, and z-score of the last N bars.
    """
    def __init__(self, mean, std, z, ts_event, ts_init):
        self.mean = mean
        self.std = std
        self.z = z
        self._ts_event = ts_event
        self._ts_init = ts_init

# --------------------
# 2. Strategy Config
# --------------------
class UVXYConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type_1day: BarType
    n_bars: int = 20
    threshold: float = 2.0
    trading_size: int = 100

# --------------------
# 3. Strategy
# --------------------
class UVXYStrategy(Strategy):
    def __init__(self, config: UVXYConfig):
        super().__init__(config=config)
        self.position = None
        self.z = 0.0

    def on_start(self):
        self.subscribe_bars(self.config.bar_type_1day)
        self.subscribe_data(DataType(ZFromNBars))

    def on_bar(self, bar: Bar):
        # Compute statistics on last N bars
        bars = self.cache.bars(self.config.bar_type_1day)[: self.config.n_bars]
        closes = np.array([float(b.close) for b in bars])
        mean = closes.mean()
        std = closes.std()
        z = (float(bar.close) - mean) / std if std else 0.0

        # Publish custom data
        data = ZFromNBars(mean, std, z, bar.ts_event, bar.ts_init)
        self.publish_data(DataType(ZFromNBars), data)

    def on_data(self, data):
        if isinstance(data, ZFromNBars):
            self.z = data.z
            self._try_enter()
            self._try_exit()

    def _try_enter(self):
        if self.z >= self.config.threshold and (not self.position or self.position.side != PositionSide.SHORT):
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.SELL,
                quantity=Quantity.from_int(self.config.trading_size)
            )
            self.submit_order(order)

    def _try_exit(self):
        if self.z < 0 and self.position and self.position.side == PositionSide.SHORT:
            self.close_position(self.position)

    def on_end(self):
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars()

# --------------------
# 4. Backtest Setup
# --------------------
if __name__ == "__main__":
    # 4.1 Instrument & BarType
    instrument = InstrumentId(symbol=Symbol("UVXY"), venue=Venue("NASDAQ"))
    UVXY_ETF_1DAY_BARTYPE = BarType.from_str(f"{instrument}-1-DAY-LAST-EXTERNAL")

    # 4.2 Config & Engine
    strat_cfg = UVXYConfig(
        instrument_id=instrument,
        bar_type_1day=UVXY_ETF_1DAY_BARTYPE,
        n_bars=20,
        threshold=2.0,
        trading_size=100
    )
    engine_cfg = BacktestEngineConfig(
        trader_id=TraderId("BACKTEST-UVXY1DAY-001"),
        strategies=[
            ImportableStrategyConfig(
                strategy_path="__main__:UVXYStrategy",
                config_path="__main__:UVXYConfig",
                config = {
                    "instrument_id": instrument,
                    "bar_type_1day": UVXY_ETF_1DAY_BARTYPE,
                    "trading_side": 50_000_000,
                    "threshold": 2,
                    "n_bars": 20
                }
            )
        ],
        logging=LoggingConfig(log_level="DEBUG")
    )
    engine = BacktestEngine(config=engine_cfg)

    # 4.3 Venue & Account
    engine.add_venue(
        venue=Venue("NASDAQ"),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, USD)],
        base_currency=USD,
        default_leverage=Decimal(1)
    )

    # 4.4 Instrument
    from nautilus_trader.model.instruments import Equity
    eq = Equity(
        instrument_id=instrument,
        raw_symbol=Symbol("UVXY"),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0
    )
    engine.add_instrument(eq)

    # 4.5 Data (example using yfinance)
    import yfinance as yf
    df = yf.download("UVXY", start="2020-01-01", end="2025-01-01")
    df = df.rename(columns={"Open":"open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume"})
    df = df / 1e9  # scale example
    wrangler = BarDataWrangler(UVXY_ETF_1DAY_BARTYPE, eq)
    bars = wrangler.process(df)
    engine.add_data(bars)

    # 4.6 Run
    engine.run()
