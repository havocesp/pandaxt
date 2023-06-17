"""Core module."""
import functools
import logging
import os
import pathlib
import sys
import warnings
from builtins import bool as Bool, int as Int, float as Float
from collections import OrderedDict
from typing import Text, Dict, List, Union as U, NoReturn, Tuple, Optional as Opt, Set

import ccxt
import pandas as pd
import tulipy
from cctf import Symbol, Tickers, Currency, Balance, Wallet, Markets, Market, Ticker, Limit, Symbols, Currencies
from diskcache import Cache
# from pandas.errors import SettingWithCopyWarning

from pandaxt.exceptions import TimeframeError, SymbolError, CurrencyError, SideError
from pandaxt.utils import load_dotenv, magic2num  # , check_symbol
from tulipy.lib import InvalidOptionError

warnings.simplefilter(action='ignore')

# noinspection PyUnusedName
pd.options.display.precision = 8
# noinspection PyUnusedName
pd.options.display.max_colwidth = None
# noinspection PyUnusedName
pd.options.display.float_format = lambda n: f'{n:.8f}'.rstrip(
    '0.') if n < 0.0 else f'{n:.8f}'

_OHLC_FIELDS = ['date', 'open', 'high', 'low', 'close', 'volume']
_DATA_DIR = pathlib.Path.home().joinpath('.local', 'share')

__all__ = ['PandaXT']

logging.basicConfig(style='{', level=logging.INFO,
                    format='[{created:.0f}] {msg}')
log = logging.getLogger('PandaxT:core')


def _get_version(exchange) -> Opt[Text]:
    """Search supplied exchange name and return latest 'ccxt' supported version.

    :param Text exchange: exchange name to check for.
    :return: the last API id for supplied exchange.
    """
    try:
        result = [ex for ex in ccxt.exchanges if ex.rstrip(
            "123456789") == exchange][-1]
    except IndexError:
        raise ValueError(
            f" - [ERROR] '{str(exchange)}' exchange is not supported.")
    return result


class PandaXT:
    """A "ccxt" exchanges wrapper class over Pandas lib."""

    def __init__(self, exchange, api_key=None, secret=None, user_agent=None, clear_cache=False, default_type=None, margin_mode=None, isolated_symbol=None):
        """Class constructor.

        >>> markets = PandaXT('binance').markets
        >>> isinstance(markets, Markets)
        True
        >>> market = markets.get('BTC/USDT')  # type: Market
        >>> isinstance(market, Market)
        True
        >>> isinstance(market.precision.price, int)
        True

        :param Text exchange: a ccxt lib supported exchange
        :param Text api_key: API key
        :param Text secret: API secret
        :param U[Bool, Text] user_agent:  if True, exchange API keys will be load from "$HOME/.env" file.
        :param Bool clear_cache:  if True, current cache will be ignored and overwrite.
        """
        # for cache purposes
        self._basemarkets = None
        self._currencies = None
        self._default_type = str(default_type or 'spot').lower()
        self._symbols = None
        self._isolated_symbol = isolated_symbol
        if str(margin_mode).lower() in ('isolated', 'cross'):
            if isolated_symbol:
                self._margin_mode = 'ISOLATED'
            else:
                raise ValueError('Isolated margin needs a symbol.')
        else:
            self._margin_mode = 'CROSS'
        # else:
            # raise ValueError(f'Invalid margin mode {margin_mode}. Supported modes are: CROSS, ISOLATED.')

        self._markets = None

        if clear_cache:
            self._cache.clear()

        exchange = _get_version(exchange)
        # assert exchange is not None, f'"{exchange}" not supported'
        api = getattr(ccxt, exchange)
        settings = dict(timeout=25000, enableRateLimit=True,
                        fetchTickersErrors=False, defaultType=self._default_type)

        if user_agent or False:
            if isinstance(user_agent, bool):
                settings.update(userAgent=api.userAgents[1])
            else:
                settings.update(userAgent=str(user_agent))

        if not all((api_key, secret)):
            load_dotenv()
            field_name = exchange.upper().strip(' _012345')
            key_field, secret_field = f'{field_name}_KEY', f'{field_name}_SECRET'
            api_key = os.environ.get(key_field)
            secret = os.environ.get(secret_field)

        if api_key and secret:
            settings.update(apiKey=api_key, secret=secret)

        if exchange == 'binance':
            # noinspection PyUnresolvedReferences
            settings.update(adjustForTimeDifference=True,
                            parseOrderToPrecision=True)
            if self._margin_mode:
                settings.update(defaultMarginMode=self._margin_mode.upper())

        self._api = api(settings)
        try:
            self._api.load_markets()
        except ccxt.ExchangeError as err:
            raise err
            if exchange == 'binance':
                log.warning(
                    'Load markers method failure. Changing to Binance US to try to solve the problem ..')
                exchange = 'binanceus'
                api = getattr(ccxt, exchange)
                self._api = api(settings)
                self._api.load_markets()
            else:
                raise err
        self._load_markets()

    def _load_markets(self) -> Markets:
        """Markets metadata cache handler.

        >>> isinstance(PandaXT('binance')._load_markets(), Markets)
        True

        :return: markets metadata as "Markets" instance.
        """
        if self._markets is not None:
            data = self._markets
        else:
            self._cache_dir = _DATA_DIR / 'pandaxt' / self.id
            self._cache_dir.mkdir(exist_ok=True, parents=True)
            self._cache = Cache(str(self._cache_dir))
            data = self._cache.get('markets', dict())
            if len(data) == 0:
                data = self._api.load_markets()
                data = {k: {x: y for x, y in v.items() if y}
                        for k, v in data.items()
                        if v}
                self._cache.set('markets', data, (60.0 ** 2.0) * 6.0)
        return Markets(**data)

    @property
    def id(self) -> Text:
        """Exchange unique reference (also know as ID).

        >>> PandaXT('binance').id
        'binance'

        :return: exchange unique reference.
        """
        return self._api.id

    @property
    def timeframes(self) -> List[Text]:
        """Return valid exchange timeframes as list.

        >>> '15m' in PandaXT('binance').timeframes
        True

        :return: valid exchange timeframes.
        """
        items = self._api.timeframes.items()
        od = OrderedDict(sorted(items, key=lambda x: x[1]))
        return list(od.keys())

    @property
    def delisted(self) -> Symbols:
        """Returns delisted symbols (active -> False)

        >>> binance = PandaXT('binance')
        >>> delisted = binance.delisted
        >>> isinstance(delisted, list)
        True

        :return: return delisted symbols
        """
        return Symbols(*[m
                         for m in self.markets
                         if hasattr(m, 'active') and not m.active])

    @property
    def name(self) -> Text:
        """Exchange long name.

        :return: exchange long name.
        """
        return getattr(self._api, 'name')

    @property
    def symbols(self) -> Symbols:
        """Get all supported symbols by exchange as "Symbol" list."""
        if self._symbols is None:
            self._symbols = sorted(
                [Symbol(m) for m in self.markets if m not in self.delisted])

        return Symbols(self._symbols)

    @property
    def base_markets(self) -> Currencies:
        """Get exchange base markets currencies as "Currency" list."""
        if self._basemarkets is None:
            self._basemarkets = sorted(list({s.quote for s in self.symbols}))
        return Currencies(self._basemarkets)

    @property
    def margin_pairs(self) -> Currencies:
        """Get exchange base markets currencies as "Currency" list."""
        if self._basemarkets is None:
            self._basemarkets = sorted(list({s.active for s in self.markets}))
        return Currencies(self._basemarkets)

    @property
    def currencies(self) -> Currencies:
        """Get supported currencies by exchange as "Currency" list.

        >>> exchange = PandaXT('binance')
        >>> len(exchange.currencies) > 0
        True

        :return: all active currencies supported by exchange.
        """
        # Initialize markets, symbols, currencies and basemarkets
        if self._currencies is None:
            self._currencies = [s.base for s in self.symbols]
            self._currencies = self._currencies + list(self.base_markets)
            self._currencies = sorted(list(set(self._currencies)))

        return Currencies(self._currencies)

    def get_market_precision(self, symbol, precision_type=None) -> Int:
        """Get precision set by exchange for a market.

        >>> PandaXT('binance').get_market_precision('MATIC/USDT')

        :param symbol: symbol of the market where precision will be return.
        :param precision_type: accepted types: "price", "amount", "cost", "base", "quote" (default "price")
        :return: precision length for the supplied market symbol.
        """
        market: Market = self.markets.get(Symbol(symbol))
        precision = market.precision
        return getattr(precision, precision_type or 'price')

    @property
    def markets(self) -> Markets:
        """Get all exchange markets metadata.

        >>> exchange = PandaXT('binance')
        >>> markets = exchange.markets
        >>> isinstance(markets, Markets) and len(markets) > 1
        True

        :return Dict: all exchange markets metadata.
        """
        if self._markets is None:
            self._markets = self._load_markets()
        return self._markets

    def parse_timeframe(self, timeframe) -> U[Int, NoReturn]:
        """Get amount of seconds for a supplied string formatted timeframe.

        :param timeframe: a supported and valid string formatted timeframe (example: "15m")
        :return: None if timeframe is not supported otherwise, amount of seconds for supplied timeframe.
        """
        if hasattr(self._api, 'parse_timeframe'):
            return self._api.parse_timeframe(timeframe)

    def get_timeframe(self, timeframe) -> Text:
        """Timeframe sanitizer.

        >>> PandaXT('binance').get_timeframe(15)
        '15m'

        :param timeframe: timeframe to sanitize.
        :type timeframe: Text or Int
        :return str: sanitize timeframe.
        """
        timeframe = str(timeframe)
        if timeframe.isdigit() or not timeframe[-1].isalpha():
            timeframe = f'{timeframe}m'
        if timeframe not in self.timeframes:
            raise TimeframeError(timeframe, exchange=self.name)
        else:
            return timeframe

    # @check_symbol
    def get_ohlc(self, symbol, timeframe='15m', limit=25) -> pd.DataFrame:
        """Get OHLC data for specific symbol as pandas DataFrame type.

        :param Text symbol: symbol name use at ohlc data request.
        :param Text timeframe: an exchange supported timeframe.
        :param int limit: max rows limit.
        :return pd.DataFrame: data-frame with: open, high, low, close, volume, qvolume columns and 'date' as index.
        """

        if Symbol(symbol) not in self.symbols:
            # print(symbol, symbol in self.symbols, len(self.symbols))
            raise SymbolError(symbol, exchange=self.name)

        data = self._api.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=_OHLC_FIELDS)
        df['qvolume'] = df['volume'] * df['close']
        df.index = pd.to_datetime(df.pop('date') // 1000, unit='s')
        df.name = 'date'
        return df

    def get_currency(self, currency) -> Currency:
        """Currency name sanitizer."""
        if currency:
            if currency in self._api.commonCurrencies:
                currency = self._api.commonCurrencies.get(currency)
            if currency not in self.currencies:
                log.debug(
                    f'Currency {str(currency)} is not supported by exchange.')
            return Currency(currency)

    def altname(self, name) -> U[Currency, Symbol]:
        """Retrieve alternative currency or symbol name used in a specific exchange.

        >>> PandaXT('binance').altname('YOYOW')
        'YOYOW'
        >>> PandaXT('binance').altname('YOYOW/BTC')
        (Symbol:YOYOW/BTC)
        >>> PandaXT('binance').altname('YOYOW/BTC')
        (Symbol:YOYOW/BTC)

        :param name: currency or symbol name to check for alternative name.
        :type name: Text or Currency or Symbol
        :return: currency alternative name as Currency or Symbol instance.
        :rtype: Currency or Symbol
        """
        _symbol = str(name).upper()
        if '/' in _symbol.strip('/'):
            # print(_symbol)
            base, quote = _symbol.split('/')
            assert quote in self.base_markets, f'{quote} is not a valid base market.'
            base = self.get_currency(base)
            s = self.get_currency(base) + Currency(quote)
            s.price2precision = functools.partial(self.price2precision, s)
            s.cost2precision = functools.partial(self.cost2precision, s)
            s.amount2precision = functools.partial(self.amount2precision, s)
            return s
        else:
            return self.get_currency(_symbol)

    def cost2precision(self, symbol, cost, to_str=True) -> U[Float, Text]:
        """Return cost rounded to symbol precision exchange specifications.

        :param symbol: a valid exchange symbol.
        :type symbol: Text or Symbol
        :param Float cost: cost to be rounded.
        :param Bool to_str: True to return result as str, otherwise result will be returned as float
        :return float: cost rounded to specific symbol exchange specifications.
        """
        result = self._api.cost_to_precision(symbol, cost)
        return result if to_str else float(result)

    def amount2precision(self, symbol, amount, to_str=True) -> U[Float, Text]:
        """Return amount rounded to symbol precision exchange specifications.

        :param symbol: a valid exchange symbol.
        :type symbol: Text or Symbol
        :param Float amount: amount to be rounded.
        :param Bool to_str: True to return result as str, otherwise result will be returned as float
        :return: amount rounded to specific symbol exchange specifications.
        :rtype: float or Text
        """
        result = self._api.amount_to_precision(symbol, amount)
        return result if to_str else float(result)

    def price2precision(self, symbol, price, to_str=True) -> U[Float, Text]:
        """Return price rounded to symbol precision exchange specifications.

        :param symbol: a valid exchange symbol.
        :type symbol: Text or Symbol
        :param price: price to be rounded.
        :param Bool to_str: True to return result as str, otherwise result will be returned as float
        :return float: price rounded to specific symbol exchange specifications.
        """
        result = self._api.price_to_precision(symbol, price)
        return result if to_str else float(result)

    def get_orderbook(self, symbol, limit=5) -> pd.DataFrame:
        """Get order book data for a symbol.

        >>> book_entries = PandaXT('binance').get_orderbook('MDA/BTC', 5)
        >>> isinstance(book_entries['ask'].values[-1], float)
        True

        :param symbol: a valid exchange symbol.
        :type symbol: Text or Symbol
        :param int limit: limit entries returned to "limit" value.
        :return: DataFrame type with order book data for "symbol".
        :rtype: pd.DataFrame
        """

        raw = self._api.fetch_order_book(symbol, limit=int(limit))

        columns = ['ask', 'ask_vol', 'bid', 'bid_vol']
        data = [ask + bid for ask, bid in zip(raw['asks'], raw['bids'])]
        df = pd.DataFrame(data, columns=columns)
        return df.round({'ask': 8, 'bid': 8, 'ask_vol': 0, 'bid_vol': 0})

    def get_tickers(self, symbols, sorted_by=None) -> U[Ticker, Tickers]:
        """Get tickers dict with symbol name as keys for all symbols specified as positional args.

        >>> exchange = PandaXT('binance')
        >>> ticker = exchange.get_tickers('ADA/BTC', sorted_by='percentage')
        >>> isinstance(ticker, Ticker)
        True

        :param symbols: list of valid exchange symbols.
        :param sorted_by: a valid ticker field like percentage, last, ask, bid, quoteVolume, ...
        :return: Ticker or Tickers instance with returned data.
        :rtype: Ticker or Tickers
        """
        result = list()

        if isinstance(symbols, (Symbols, List, Tuple, Set, str)) and len(symbols):
            symbols = [symbols] if isinstance(symbols, str) else list(symbols)

        # for s in map(self.altname, symbols):
        #     if s not in self.markets:
        #         log.debug(f'Symbol {s or "NULL"} is not listed in {self.name} exchange.')
        #         continue
        try:
            raw = self._api.fetch_tickers(
                symbols=[str(s) for s in symbols if self.altname(s) in self.markets])
            if len(raw) > 0:
                if str(sorted_by) in list(raw.values())[0].keys():
                    raw = OrderedDict(
                        sorted(raw.items(), key=lambda k: k[1][sorted_by], reverse=True))
                result = Tickers(**raw)
                result = result[symbols[0]] if len(symbols) == 1 else result
        except ccxt.ExchangeError as err:
            print(str(err))
        return result

    def get_indicators(self, indicators, symbol, timeframe='15m', limit=25, **kwargs) -> Dict[Text, pd.Series]:
        """Get technical indicators value for a symbol.

        :param Dict indicators: indicators and their params as dict (params ara mandatory, there is no default values).
        :param Text symbol: a valid exchange symbol.
        :param Text timeframe: an exchange valid timeframe (default 15m).
        :param Int limit: a valid exchange limit for returned rows (check exchange official API)
        supplied as a param / value dict instance also. Example: "{'roc': {'period': 9}}"
        :param kwargs: if "ohlc" is set with OHLC data (DataFrame) it will be use for value calculations.
        :return: dict type with indicators name/value pairs.
        """
        indicators = {k.lower(): v for k, v in indicators.items()}
        symbol = Symbol(symbol)
        return_value = OrderedDict.fromkeys(indicators.keys())
        supported_ti = [_a for _a in dir(tulipy.lib) if _a[0].islower()]

        functions = OrderedDict({i: getattr(tulipy, i)
                                 for i in indicators if i in supported_ti})

        data = kwargs.get('ohlc', self.get_ohlc(
            symbol, timeframe=timeframe, limit=limit))

        for n, fn in functions.items():
            inputs = ['close' if i in 'real' else i for i in fn.inputs]
            indicator_params = dict()
            if len(fn.options):
                options = [opt.replace(' ', '_') for opt in fn.options]
                indicator_params = indicators.get(n)
                indicator_params = {k: v
                                    for k, v in indicator_params.items()
                                    if k in options}

            try:
                raw = fn(*data[inputs].T.values, **indicator_params)
                di = data.index
                if n in 'roc':
                    raw = raw * 100.0
                sr = pd.Series(raw, name=n.upper())
                sr.index = di.values[-len(sr):]
                return_value[n] = sr.copy(True)

            except InvalidOptionError as err:
                print(str(err))

        return dict(ohlc=data, **{k: return_value[k.lower()] for k in indicators})

    def create_market_order(self, symbol, side, amount=None) -> Dict:
        """Create a market order order.

        :param Text symbol: a valid exchange symbol.
        :param Text side: accepted values: "buy", "sell"
        :param float amount: amount used in order creation.
        :return: order creation result data as dict.
        """
        return self.create_order(symbol=symbol, side=side, order_type='market', amount=amount)

    def buy(self, symbol, amount=None, price=None) -> Dict:
        """Create buy order.

        :param Text symbol: a valid exchange symbol.
        :param float amount: amount to buy or None to auto-fill
        :param float price: buy price or None to auto-fill
        :return Dict: order creation result data as dict.
        """
        return self.create_order(symbol=symbol, order_type='limit', side='buy', amount=amount, price=price)

    def create_order(self, symbol, side, order_type=None, amount=None, price=None) -> Dict:
        """Create a new order.

        :param Text symbol: symbol to be use for order creation.
        :param Text side: order side: 'sell' or 'buy'
        :param Text order_type: order type (default 'limit')
        :param float amount: amount used in order creation.
        :param float price: price used in order creation.
        :return: order result info as dict.
        """
        symbol = Symbol(symbol)
        response = dict()

        if symbol not in self.symbols:
            raise SymbolError(f'Invalid symbol: {symbol}')

        if side not in ['buy', 'sell']:
            raise SideError(side)

        currency = symbol.quote if side in 'buy' else symbol.base
        balance_field = 'free' if side in 'buy' else 'total'
        ticker_field = 'ask' if side in 'buy' else 'bid'

        amount = magic2num(
            amount or self.get_balances(balance_field).get(
                currency, 0.0))

        if amount > 0.0:
            price = magic2num(price or self.get_tickers(
                symbol).get(ticker_field))
            if side in 'buy':
                amount = amount / price
            try:
                response = self._api.create_order(symbol=symbol,
                                                  type=order_type or 'limit',
                                                  side=side,
                                                  amount=amount,
                                                  price=price)
            except ccxt.InvalidOrder as err:
                print(f' - [ERROR] {str(err)}', file=sys.stderr)
                response = dict()
        return response

    def sell(self, symbol, amount=None, price=None) -> Dict:
        """Create sell order.

        :param Text symbol: a valid exchange symbol.
        :param Float amount: amount to sell or None to auto-fill
        :param Float price: sell price or None to auto-fill
        :return dict: order creation result data as dict.
        """
        return self.create_order(symbol=symbol, order_type='limit', side='sell', amount=amount, price=price)

    def get_balances(self, field=None, tradeables_only=True, fiat=None, base_market=None) -> Wallet:
        """Get balances.

        >>> balances = PandaXT('binance').get_balances('total')
        >>> isinstance(balances, Wallet)
        True

        :param Text field: accepted values: if None all balances will be loaded, (values; "total", "used", "free")
        :param Bool tradeables_only:
        :param fiat:
        :param base_market:
        :return: positive balances.
        """

        def is_zero(v) -> U[Float, Dict]:
            if isinstance(v, float):
                return v <= 0.0
            elif isinstance(v, dict):
                return v.get(field or 'total', 0.0) <= 0.0
            else:
                return 0.0

        def is_tradeable(currency, _tickers, balance, base_market=None) -> Bool:
            """Check if a currency balance is over minimum tradeable amount for a base market.

            :param Text currency:
            :param Tickers _tickers:
            :param balance:
            :type balance: Dict or Float
            :param Text base_market:
            :return: True if currency is tradeable
            """
            if currency in self.base_markets:
                return True

            base_market = Currency(base_market or 'BTC')
            symbol: Symbol = self.altname(currency) + base_market

            market: Market = self.markets.get(symbol, False)
            if not market:
                return False
            if symbol not in _tickers:
                return False
            # min amount in quote currency
            limits: Limit = market.limits
            quote_min_amount = limits.amount['min']

            ticker: Ticker = _tickers[symbol]
            if currency == 'BTC' and 'USD' in base_market:
                last = 1.0 / ticker.last
            else:
                last = ticker.last
            # converting min_amount to base currency

            base_min_amount = last * quote_min_amount
            # subtracting a 0.01% fee
            base_min_amount = base_min_amount * 0.999
            if isinstance(balance or [], dict):
                balance = balance.get('total', 0.0)
            else:
                balance = balance or 0.0
            return balance > base_min_amount

        try:
            base_market = Currency(base_market or 'BTC')
            data = self._api.fetch_balance()

            if 'info' in data:
                del data['info']

            data: dict = data.pop(field or 'total')

            data = {str(k): v.get(field or 'total', v) if isinstance(v, dict) else v
                    for k, v in data.items()
                    if not is_zero(v.get(field or 'total', v) if isinstance(v, dict) else v)}

            if tradeables_only:
                symbols = {
                    Currency(c) + base_market for c in data if c not in self.base_markets}

                tickers = self.get_tickers(symbols)

                data = {str(k): v
                        for k, v in data.items()
                        if is_tradeable(k, tickers, v, base_market)}

            wallet = Wallet(**data)
            if any(f in str(fiat or '').lower() for f in ['eur', 'usd']):
                fiat = str(fiat or '').lower()
                acum = 0
                for k, v in Wallet(**data).items():
                    if fiat == 'eur':
                        acum += v.to_eur
                    elif fiat == 'usd':
                        acum += v.to_usd
                wallet.update({fiat.upper(): acum})
            return wallet
        except ccxt.NetworkError as err:
            print('PandaXT::get_ohlc', str(err))

    def get_balance(self, currency, field=None, base_market=None) -> Balance:
        """Get balance for a currency.

        >>> PandaXT('binance').get_balance('STORJ', 'total')

        :param Text currency: a valid exchange currency.
        :param Text field: accepted values: total, used, free
        :return: currency with balance amount as float.
        """
        currency = self.altname(str(currency))

        if currency not in self.currencies:
            raise CurrencyError(currency)

        if field and field in ['total', 'free', 'used']:
            field = field
        else:
            field = 'total'

        balance_data = self.get_balances(field=field, base_market=base_market) or Balance(
            **{'currency': currency, field: 0.0})
        if currency not in balance_data:
            raise ccxt.InsufficientFunds()
        else:
            return balance_data[currency]

    def get_user_trades(self, symbol, side=None, limit=25) -> pd.DataFrame:
        """Get user trades filter by symbol.

        :param Text symbol: a valid exchange symbol
        :param Text side: "buy" or "sell"
        :param int limit: a valid limit for rows return (please, refer to official exchange API manual for details)
        :return pd.DataFrame: user trades as pandas DataFrame type.
        """
        symbol = str(symbol).upper()
        if symbol not in (self.symbols, self.altname(symbol) or ''):
            raise SymbolError(symbol)
        else:
            symbol = Symbol(
                symbol) if symbol in self.symbols else self.altname(symbol)

        trades = self._api.fetch_my_trades(symbol, limit=limit)
        if trades:
            trades = [{k: v for k, v in t.items() if k not in 'info'}
                      for t in trades]
            for idx, t in enumerate(trades.copy()):
                trades[idx].update(total_cost=trades[idx]['fee']['cost'])
                del trades[idx]['fee']
            trades = pd.DataFrame(trades)
            trades['real_cost'] = trades['cost'] + \
                (trades['cost'] * trades['price'])
            # TODO: not totally true so revise it
            trades['real_price'] = trades['price'] * 1.001
            trades['real_amount'] = trades['real_cost'] * trades['price']
            if str(side).lower() in ['buy', 'sell']:
                trades = trades.query(f'side == "{str(side).lower()}"')

            return trades.sort_index(ascending=False)

    def get_trades(self, symbol, side=None, limit=25) -> pd.DataFrame:
        """Get latest user trades for a symbol.

        :param Text symbol: a valid exchange symbol
        :param Text side: "buy" or "sell"
        :param int limit: a valid limit for rows return (please, refer to official exchange API manual for details)
        :return: latest trades for a symbol.
        """
        symbol = self.altname(symbol)
        trades = self._api.fetch_trades(symbol, limit=limit)
        if trades:
            trades = [{k: v for k, v in t.items() if k not in 'info' and v}
                      for t in trades]
        trades = pd.DataFrame(trades).set_index('timestamp')
        if str(side).lower() in ['buy', 'sell']:
            trades = trades.query(f'side == "{str(side).lower()}"')
        return trades.sort_index(ascending=False)

    def get_cost(self, symbol, **kwargs) -> Float:
        """FIXME do not work well sometimes giving a bad result by unknown reason.

        Get weighted average (from buy trades data) cost for a symbol.

        >>> api = PandaXT('binance')
        >>> symbol_cost = api.get_cost('AGI/BTC')
        >>> symbol_cost
        True

        :param U[Symbol,Text] symbol: a valid exchange symbol.
        :param kwargs: accepted keys are: balance (Balance) and trades (pd.DataFrame)
        :return: cost calculation result as float type.
        """
        symbol = self.altname(symbol)
        if isinstance(symbol or 0, Currency):
            symbol = symbol + Currency(kwargs.get('base_market', 'BTC'))

        base, quote = symbol.parts

        # reuse provided balance and trades data (for rate limit save)
        if 'balance' in kwargs:
            cached = kwargs.get('balance')  # type: Balance
            if base not in cached:
                raise ValueError(f'No {base} balance found on wallet.')
            balance = cached[base] if isinstance(cached, Wallet) else cached
        else:
            balance = self.get_balance(
                currency=base, field='total', base_market=quote)  # type: Balance

        total_balance = balance.total if balance and hasattr(
            balance, 'total') else balance

        if total_balance > 0.0:
            trades = kwargs.get('trades', [])
            if not trades:
                trades = self.get_user_trades(symbol, side='buy')
            elif not isinstance(trades, pd.DataFrame):
                trades = pd.DataFrame(trades)
            if trades['order'].isna().all():
                trades['order'].update(trades['id'])

            # group-by operations per column
            columns_op = {'amount': 'mean',
                          'price': 'mean',
                          'cost': 'mean',
                          'timestamp': 'mean'}

            trades = trades.groupby('order').agg(columns_op).sort_index(
                ascending=False)  # type: pd.DataFrame
            trades = trades[['price', 'amount']].reset_index(drop=True)

            for index, price, amount in trades.itertuples():
                if round(total_balance - amount, 8) <= 0.0:
                    if round(total_balance - amount, 8) != 0.0:
                        prices = trades[:index + 1]['price'].values
                        amounts = trades[:index + 1]['amount'].values
                        amounts[-1] = total_balance
                    else:
                        prices, amounts = trades[:index + 1].T.values
                    return round(pd.np.average(prices, weights=amounts), 10)
                else:
                    total_balance -= amount

    def get_order_status(self, order_id, symbol=None) -> Text:
        """Get order status by order_id.

        :param Text order_id: a valid order id.
        :param Text symbol: a valid exchange market
        :return: order status as str. Possible values are: "closed",  "canceled", "open"
        """
        return self._api.fetch_order_status(order_id, symbol=symbol)

    def get_open_orders(self, symbol=None, order_id=None) -> List[Dict[Symbol, Dict]]:
        """Get open orders.md for a symbol.

        :param Text symbol: symbol used in opened orders.md server query.
        :param order_id: just return data for a specific order.
        :type order_id: Text or int
        :return List: list of open orders.md for specific symbol.
        """

        raw = self._api.fetch_open_orders(symbol or order_id)
        if isinstance(raw or 0, list) and len([r for r in raw if r]):
            return [{Symbol(k): v for k, v in r.items() if k not in 'info'} for r in raw]
        else:
            return list()

    def get_profit(self, currency) -> U[Tuple, Float]:
        """Returns current profit for a currency and its weighted average buy cost.

        :param Text currency: a valid currency to use at profit and cost calc
        :return: current profit and weighted average buy cost as a tuple
        """
        currency = Currency(str(currency))
        btc_symbol = currency + Currency('BTC')  # type: Symbol
        balance = self.get_balance(currency)

        if balance.used > 0.0:
            cost = self.get_cost(symbol=btc_symbol)
            return balance.to_btc - (cost * balance.total)
        else:
            return 0.0, 0.0

    def get_withdrawals(self):
        """Get withdrawals info.

        :return:
        """
        if self._api.has['fetchWithdrawals']:
            result = self._api.fetch_withdrawals()
            return [{k: v for k, v in r.items() if k != 'info'} for r in result]
        else:
            raise ccxt.NotSupported(
                'Withdrawals not supported by this exchange.')

    def get_transactions(self):
        """Get transactions info.

        :return:
        """
        if self._api.has['fetchTransactions']:
            result = self._api.fetch_transactions()
            return [{k: v for k, v in r.items() if k != 'info'} for r in result]
        else:
            raise ccxt.NotSupported(
                'Transactions not supported by this exchange.')

    def get_deposits(self):
        """Get deposits info.

        :return:
        """
        if self._api.has['fetchDeposits']:
            result = self._api.fetch_withdrawals()
            return [{k: v for k, v in r.items() if k != 'info'} for r in result]
        else:
            raise ccxt.NotSupported('Deposits not supported by this exchange.')

    def cancel_order(self, symbol: U[Bool, Text], last_only=False) -> List:
        """Cancel symbols open orders.md for a symbol.

        :param symbol: the symbol with open orders.md.
        :type symbol: Bool or Symbol
        :param Bool last_only: if True, only last order sent will be cancelled.
        :return: list of dict with data about cancellations.
        """
        symbol = Symbol(symbol)
        pending_orders = self.get_open_orders(symbol)

        if len(pending_orders):
            if last_only:
                return self._api.cancel_order(pending_orders[-1]['id'], str(symbol))
            else:
                canceled_orders = list()
                for p in pending_orders:
                    return_value = self._api.cancel_order(p['id'], symbol)

                    if return_value and return_value.get('status', '') in 'cancel':
                        canceled_orders.append(
                            {k: v for k, v in return_value.items() if v})
                    else:
                        self._api.cancel_order(p['id'], symbol)
                return canceled_orders

    def __str__(self) -> Text:
        """PandaXT instance as "str" type representation.

        >>> str(PandaXT('binance'))
        'binance'

        :return Text: PandaXT instance as "str" type representation.
        """
        return self.id

    def __repr__(self):
        """PandaXT instance as "str" type representation.

        >>> PandaXT('binance')
        binance

        :return str: PandaXT instance as "str" type representation.
        """
        return f'{type(self).__name__}<{self.id}>'

    def __contains__(self, item) -> Bool:
        """Check if a symbol or currency is supported by exchange.

        >>> exchange = PandaXT('binance')
        >>> Currency('ETH') in exchange
        True
        >>> Currency('ETH/BTC') in exchange
        True
        >>> Currency('MyCOIN') in exchange
        False

        :param item: currency or symbol for supports checking on exchange.
        :type item: Text or Currency or Symbol
        :return bool: True is item is supported, otherwise False.
        """
        return str(item) in self.markets or str(item) in map(str, self.currencies)
