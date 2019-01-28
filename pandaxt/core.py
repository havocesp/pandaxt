# -*- coding:utf-8 -*-
"""Core module."""
import collections
import functools
import os
import pathlib
import warnings
from collections import OrderedDict, UserDict, Iterable

import ccxt
import pandas as pd
import tulipy
from cctf import Symbol, Tickers, Currency, Balance, Wallet, Markets, Market, Ticker
from diskcache import Cache
from pandas.core.common import SettingWithCopyWarning
# from cctf import Market, Symbol, Currency, Ticker, Markets, Wallet, Tickers
from pandaxt.decorators import retry
from pandaxt.utils import load_dotenv, magic2num, get_tor_session, error  # , find_nearest

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# log = Logger('PandaXT', 'INFO')
_COLORS = ("red", "green", "yellow", "blue", "magenta", "cyan")

pd.options.display.precision = 8
pd.options.display.max_colwidth = -1
pd.options.display.float_format = lambda n: '{:.8f}'.format(n).rstrip('0.') if n < 0.0 else '{:.8f}'.format(n)

_OHLC_FIELDS = ['date', 'open', 'high', 'low', 'close', 'volume']
_USER_DATA_DIR = pathlib.Path.home().joinpath('.local', 'share')
_SETTINGS = dict(config=dict(timeout=25000, enableRateLimit=True, fetchTickersErrors=False))

__all__ = ['PandaXT']


def _get_version(exchange):
    """Exchange CCXT id validator an sanitizer.

    If exchange is found in CCXT supported exchange list and exchange has only one API api version, exchange value
    will be returned as is.

    If exchange has more than one API versions implemented by CCXT library, the most recent will be returned.

    And finally, if no id is found returned value is None.

    :param exchange: exchange name to check for.
    :type exchange: str
    :return: the last API id for supplied exchange.
    :rtype: str or None
    """
    return [ex for ex in ccxt.exchanges if exchange == ex.strip(' _12345')]


# noinspection PyUnusedFunction,PySameParameterValue
class PandaXT:
    """A "ccxt" exchanges wrapper class over Pandas lib."""

    def __init__(self, exchange, load_keys=True, tor=False, user_agent=None):
        """
        Constructor.

        >>> markets = PandaXT('binance').markets
        >>> isinstance(markets, Markets)
        True
        >>> market = markets.get('BTC/USDT')  # type: Market
        >>> isinstance(market, Market)
        True
        >>> isinstance(market.precision.price, int)
        True

        :param Text exchange: a ccxt lib supported exchange
        :param Bool load_keys:  if True, exchange API keys will be load from "$HOME/.env" file.
        :param Bool load_keys:  if True, exchange API keys will be load from "$HOME/.env" file.
        :param tor: if True, tor network proxy will be use.
        :param user_agent:  if True, exchange API keys will be load from "$HOME/.env" file.
        :type user_agent: Bool or Text
        """

        self._symbols = list()
        self.basemarkets = list()
        self.currencies = list()

        exchange = str(exchange).lower()

        if exchange not in ccxt.exchanges:
            raise ValueError('{} not supported'.format(exchange))

        if '{}2'.format(exchange) in ccxt.exchanges:
            exchange = '{}2'.format(exchange)
        elif '{}3'.format(exchange) in ccxt.exchanges:
            exchange = '{}3'.format(exchange)

        self._cache_dir = _USER_DATA_DIR.joinpath('pandaxt', exchange.rstrip('23'))
        self._cache_dir.mkdir(exist_ok=True, parents=True)
        self._cache = Cache(str(self._cache_dir))

        api = getattr(ccxt, exchange)
        settings = _SETTINGS.get('config')

        if user_agent is not None:
            if isinstance(user_agent, bool) and user_agent is True:
                settings.update(userAgent=api.userAgents[1])
            elif isinstance(user_agent, str) and len(user_agent):
                settings.update(userAgent=user_agent)

        if tor:
            settings.update(session=get_tor_session())

        if load_keys:
            load_dotenv()
            self.key = os.environ.get('{}_KEY'.format(exchange.upper().strip('_012345')))
            self.secret = os.environ.get('{}_SECRET'.format(exchange.upper().strip('_012345')))

            if self.key and self.secret:
                settings.update(apiKey=self.key, secret=self.secret)

        self._api = api(config=settings)

        if exchange in 'binance':
            # noinspection PyUnresolvedReferences
            self._api.load_time_difference()
            self._api.options['parseOrderToPrecision'] = True

    @classmethod
    def is_supported(cls, exchange):
        """Exchange support checker.

        >>> PandaXT.is_supported('binance')
        True

        :param str exchange: exchange name to be checked.
        """
        if isinstance(exchange or 0, str) and len(exchange) > 0:
            return any([str(exchange) == e for e in ccxt.exchanges])
        else:
            return False
    @property
    def id(self):
        """Exchange unique reference (also know as ID).

        >>> PandaXT('binance').id
        'binance'

        :return Text: exchange unique reference.
        """
        return self._api.id


    @property
    def timeframes(self):
        """Return valid exchange timeframes as list.

        >>> '15m' in PandaXT('binance').timeframes
        True

        :return List: valid exchange timeframes.
        """
        items = self._api.timeframes.items()
        od = OrderedDict(sorted(items, key=lambda x: x[1]))
        return list(od.keys())

    @property
    def delisted(self):
        """Returns delisted symbols (active -> False)

        >>> binance = PandaXT('binance')
        >>> delisted = binance.delisted
        >>> isinstance(delisted, list)
        True

        :return List: return delisted symbols
        """
        return [v for k, v in self.markets.items() if hasattr(v, 'active') and not v.active]

    @property
    def symbols(self):
        return [self.altname(s) for s in self._api.load_markets()]

    @property
    def name(self):
        """Exchange long name.

        >>> PandaXT('binance').name
        'Binance'

        :return Text: exchange long name.
        """
        return getattr(self._api, 'name')

    @property
    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def markets(self):
        """Get all exchange markets metadata.

        >>> isinstance(PandaXT('binance').markets, Markets)
        True

        :return Dict: all exchange markets metadata.
        """
        data = self._cache.get('markets')
        if data is None:
            raw = self._api.load_markets()
            raw = {k: {x: y for x, y in v.items() if y}
                   for k, v in raw.items()
                   if v}
            data = Markets(**raw)
            self._cache.set('markets', data, (60.0 ** 2.0) * 6.0)
        if not len(self._symbols):
            self._symbols = [self.altname(s) for s in sorted(data)]
            self.basemarkets = list({s.quote for s in self._symbols})
            self.currencies = list({s.base for s in self._symbols})
        return Markets(**data) if not isinstance(data, Markets) else data

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def _fetch_ohlcv(self, symbol, timeframe='15m', since=None, limit=None, params=None):
        """See `Exchange` fetch_ohlcv from ccxt lib.

        :param symbol:
        :param timeframe:
        :param since:
        :param limit:
        :param params:
        :type symbol: Text or Symbol
        :type timeframe: Text
        :type limit: int
        :type params: Dict
        :return:
        """
        if not self._api.has['fetchTrades']:
            self._api.raise_error(ccxt.NotSupported, details='fetch_ohlcv() not implemented yet')

        timeframe2seconds = self._api.parse_timeframe(timeframe) * limit * 2
        trades = self._api.fetch_trades(symbol, since=timeframe2seconds * 1000, params=params or dict())

        return self._api.build_ohlcv(trades, timeframe, since)

    def get_timeframe(self, timeframe):
        # timeframe = find_nearest(self._api.timeframes.values(), value)
        if isinstance(timeframe, int):
            timeframe = '{}m'.format(timeframe)
        # elif str(timeframe)[-1] in list('mhwMd'):
        return str(timeframe) in self.timeframes

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def get_ohlc(self, symbol, timeframe='5m', limit=25):
        """
        Get OHLC data for specific symbol as pandas DataFrame type.

        :param Text symbol: symbol name use at ohlc data request.
        :param Text timeframe: an exchange supported timeframe.
        :param int limit: max rows limit.
        :return pd.DataFrame: DataFrame with timestamps a index and columns: open, high, low, close, volume, qvolume
        """

        if not Symbol(symbol) in self.symbols:
            raise ValueError('{} not supported'.format(symbol))

        if self.id in 'cryptopia':
            data = self._fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        else:
            data = self._api.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=_OHLC_FIELDS)
        df['qvolume'] = df['volume'] * df['close']
        df.index = pd.to_datetime(df.pop('date') // 1000, unit='s')
        df.name = 'date'
        return df

    def altname(self, name):
        """Retrieve alternative currency or symbol name used in a specific exchange.

        >>> yoyo_btc = PandaXT('binance').altname('YOYO/BTC')
        >>> yoyo_btc
        'YOYOW/BTC'
        >>> yoyo_btc.price2precision(0.000140308786, to_str=False)
        0.00014031

        :param name: currency or symbol name to check for alternative name.
        :type name: Text or Currency or Symbol
        :return: currency alternative name as Currency or Symbol instance.
        :rtype: Currency or Symbol
        """
        ccc = self._api.common_currency_code
        if '/' in str(name):
            s = Symbol(name)
            s = Symbol(ccc(s.base), s.quote)
            s.price2precision = functools.partial(self.price2precision, s)
            s.cost2precision = functools.partial(self.cost2precision, s)
            s.amount2precision = functools.partial(self.amount2precision, s)
            return s
        else:
            c = ccc(str(name))
            if c == 'BSV':
                c = 'BCHSV'
            return Currency(c)

    def cost2precision(self, symbol, cost, to_str=True):
        """
        Return cost rounded to symbol precision exchange specifications.

        :param symbol: a valid exchange symbol.
        :type symbol: Text or Symbol
        :param float cost: cost to be rounded.
        :return float: cost rounded to specific symbol exchange specifications.
        """
        t = str if to_str else float
        return t(self._api.cost_to_precision(symbol, cost))

    def amount2precision(self, symbol, amount, to_str=True):
        """
        Return amount rounded to symbol precision exchange specifications.

        :param symbol: a valid exchange symbol.
        :type symbol: Text or Symbol
        :param amount: amount to be rounded.
        :param bool to_str: True to return result as str, otherwise result will be returned as float
        :return: amount rounded to specific symbol exchange specifications.
        :rtype: float or Text
        """
        t = str if to_str else float
        return t(self._api.amount_to_precision(symbol, amount))

    def price2precision(self, symbol, price, to_str=True):
        """
        Return price rounded to symbol precision exchange specifications.

        :param symbol: a valid exchange symbol.
        :type symbol: Text or Symbol
        :param price: price to be rounded.
        :param Bool to_str: True to return result as str, otherwise result will be returned as float
        :return float: price rounded to specific symbol exchange specifications.
        """
        m = self.markets.get(symbol)  # type: Market
        t = str if to_str else float
        template = '{:.@f}'.replace('@', str(m.precision.price))
        return t(template.format(float(price)))

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def get_orderbook(self, symbol, limit=5):
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

    # @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def get_tickers(self, symbols, sorted_by=None):
        """Get tickers dict with symbol name as keys for all symbols specified as positional args.

        >>> exchange = PandaXT('binance')
        >>> ticker = exchange.get_tickers("ADA/BTC", sorted_by='percentage')
        >>> isinstance(ticker, Ticker)
        True

        :param symbols: list of valid exchange symbols.
        :type symbols: Iterable
        :param sorted_by: a valid ticker field like percentage, last, ask, bid, quoteVolume, ...
        :return: Ticker or Tickers instance with returned data.
        :rtype: Ticker or Tickers
        """
        result = list()

        if isinstance(symbols, Iterable) and len(symbols):
            symbols = [symbols] if isinstance(symbols, str) else list(symbols)

            for s in map(str, symbols):
                if s not in self.markets:
                    print('Symbol {} is not listed in {} exchange.'.format(s or 'NULL', self.name))
            try:
                raw = self._api.fetch_tickers(symbols)
                # if hitbtc in self.id:
                #     raw['ask'] = raw['info']['ask']
                #     raw['bid'] = raw['info']['bid']
                #     raw['last'] = float((raw['ask'] + raw['bid'])/2.0)
                if str(sorted_by) in list(raw.values())[0].keys():
                    raw = OrderedDict(sorted(raw.items(), key=lambda k: k[1][sorted_by], reverse=True))
                result = Tickers(**raw)
                result = result[symbols[0]] if len(symbols) == 1 else result
            except ccxt.ExchangeError as err:
                print(str(err))

        return result

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def get_indicators(self, indicators, symbol, timeframe='15m', limit=25, **kwargs):
        """Get technical indicators value for a symbol.

        Indicators params should be supplied

        :param Dict indicators: indicators and their params as dict (params ara mandatory, there is no default values).
        :param Text symbol: a valid exchange symbol.
        :param Text timeframe: an exchange valid timeframe (default 15m).
        :param int limit: a valid exchange limit for returned rows (check exchange official API)
        :param indicators_params: dict instance containing indicator / params key pair where params (if any) will be
        supplied as a param / value dict instance also. Example: "{'roc': {'period': 9}}"
        :param kwargs: if "ohlc" is set with OHLC data (DataFrame) it will be use for value calculations.
        :return Dict: dict type with indicators name/value pairs.
        """
        indicator_names = indicators.keys()
        indicators = {k.lower(): v for k, v in indicators.items()}
        symbol = Symbol(symbol)
        result = OrderedDict.fromkeys(indicators.keys())
        supported_ti = [a for a in dir(tulipy.lib) if a[0].islower()]

        functions = OrderedDict({i: getattr(tulipy, i) for i in indicators if i in supported_ti})

        data = kwargs.get('ohlc', self.get_ohlc(symbol, timeframe=timeframe, limit=limit))

        for n, fn in functions.items():
            inputs = ['close' if i in 'real' else i for i in fn.inputs]
            indicator_params = dict()
            if len(fn.options):
                options = [opt.replace(' ', '_') for opt in fn.options]
                indicator_params = indicators.get(n)
                indicator_params = {k: v for k, v in indicator_params.items() if k in options}

            try:
                raw = fn(*data[inputs].T.values, **indicator_params)
                di = data.index
                if n in 'roc':
                    raw = raw * 100.0
                sr = pd.Series(raw, name=n.upper())
                sr.index = di.values[-len(sr):]
                result[n] = sr.copy(True)

            except tulipy.lib.InvalidOptionError as err:
                print(str(err))

        return {k: result[k.lower()] for k in indicator_names}

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def create_market_order(self, symbol, side):
        """Create a market order order.

        :param Text symbol: a valid exchange symbol.
        :param Text side: accepted values: "buy", "sell"
        :return Dict: order creation result data as dict.
        """
        symbol = Symbol(symbol)
        side = str(side).lower()

        if symbol not in self._symbols:
            error('Invalid symbol: {}'.format(symbol))

        if side not in ['buy', 'sell']:
            error('Invalid side: {} (accepted values: "buy", "sell")'.format(side))

        coin = quote if side in 'buy' else base
        free = self.get_balance(coin, 'free') or dict()

        return self._api.create_order(symbol, type='market', side=side, amount=magic2num(amount))

    def buy(self, symbol, amount=None, price=None):
        """
        Create buy order.

        :param symbol: a valid exchange symbol.
        :type symbol: str or Symbol
        :param float amount: amount to buy or None to auto-fill
        :param float price: buy price or None to auto-fill
        :return Dict: order creation result data as dict.
        """
        return self.create_order(symbol, order_type='limit', side='buy', amount=amount, price=price)

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def create_order(self, symbol, side, order_type=None, amount=None, price=None):
        """
        Create a new order.

        :param symbol: symbol to be use for order creation.
        :type symbol: str or Symbol
        :param Text side: order side: 'sell' or 'buy'
        :param Text order_type: order type (default 'limit')
        :param float amount: amount used in order creation.
        :param float price: price used in order creation.
        :return Dict:
        """
        symbol = Symbol(symbol)
        assert symbol in self._symbols, 'Invalid symbol: {}'.format(symbol)
        base, quote = symbol.parts
        if side in 'buy':
            currency = quote
            balance_field = 'free'
            ticker_field = 'ask'
        else:
            balance_field = 'total'
            ticker_field = 'bid'
            currency = base
        amount = magic2num(amount or self.get_balances(balance_field).get(currency, 0.0))
        if amount > 0.0:
            price = magic2num(price or self.get_tickers(symbol).get(ticker_field))
            if side in 'buy':
                amount = amount / price
            return self._api.create_order(symbol, type=str(order_type or 'limit'), side=side, amount=amount,
                                          price=price)

    def sell(self, symbol, amount=None, price=None):
        """
        Create sell order.

        :param symbol: a valid exchange symbol.
        :type symbol: str or Symbol
        :param float amount: amount to sell or None to auto-fill
        :param float price: sell price or None to auto-fill
        :return dict: order creation result data as dict.
        """
        return self.create_order(symbol, order_type='limit', side='sell', amount=amount, price=price)

    @retry(exceptions=ccxt.NetworkError)
    def get_balances(self, field=None):
        """Get balances.

        :param Text field: accepted values: if None all balances will be loaded, (values; "total", "used", "free")
        :return Wallet: positive balances.
        """
        raw = self._api.fetch_balance()
        if 'info' in raw:
            del raw['info']
        data = raw.pop(field) if field else raw  # type: dict

        def nonzero(v):
            if isinstance(v, float):
                return v > 0.0
            else:
                return v['total'] if 'total' in v else False

        return Wallet(**{str(k): v for k, v in data.items() if nonzero(v)})

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def get_balance(self, currency, field='total'):
        """Get balance for a currency.

        :param currency: a valid exchange currency.
        :type currency: Text or Currency
        :param Text field: accepted values: total, used, free
        :return Balance: currency with balance amount as float.
        """

        currency = Currency(str(currency))

        if currency not in self.currencies:
            raise ValueError('{} exchange do not support {} currency'.format(self.name.upper(), currency))
        balance_data = self.get_balances(field=field) or Balance(currency=currency, **{field: 0.0})
        # balance_data = balance_data.get(currency, balance_data)
        return balance_data[currency]

    def _calculate_fees(self, trades):
        """

        :param pd.DataFrame trades:
        :return:
        """
        fee = trades.pop('fee')
        if fee.any():
            fee_currency = pd.Series(fee.apply(lambda v: v['currency']), index=trades.index.values, name='fee_currency')
            trades['fee_currency'] = fee_currency
            trades['fee_percent'] = trades.T.apply(lambda v: 0.05 if 'BNB' in v['fee_currency'] else 0.1).T
            trades['fee_base'] = trades['fee_percent'] / 100. * trades['cost']
            trades['total'] = trades.T.apply(
                lambda v: v['cost'] - v['fee_base'] if v['side'] in 'sell' else v['cost'] + v['fee_base']).T

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def get_user_trades(self, symbol, limit=25):
        """Get user trades filter by symbol.

        :param symbol: a valid exchange symbol
        :type symbol: Text or Symbol
        :param int limit: a valid limit for rows return (please, refer to official exchange API manual for details)
        :return pd.DataFrame: user trades as pandas DataFrame type.
        """
        symbol = self.altname(symbol)

        trades = self._api.fetch_my_trades(symbol, limit=limit)
        if trades:
            trades = [{k: v for k, v in t.items() if k not in 'info'} for t in trades]
            for idx, t in enumerate(trades.copy()):
                fee_dict = trades[idx].pop('fee')
                currency = fee_dict.pop('currency')
                cost = fee_dict.pop('cost')
                trades[idx].update(fee_currency=currency, fee_cost=cost)
            trades = pd.DataFrame(trades)

            return trades.sort_index(ascending=False)

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def get_trades(self, symbol, limit=25):
        """Get latest user trades for a symbol.

        :param symbol: a valid exchange symbol
        :param limit: a valid limit for rows return (please, refer to official exchange API manual for details)
        :return pd.DataFrame: latest trades for a symbol.
        """
        symbol = self.altname(symbol)
        trades = self._api.fetch_trades(symbol, limit=limit)
        if trades:
            trades = [{k: v for k, v in t.items() if k not in 'info'} for t in trades]
        trades = pd.DataFrame(trades)
        return trades.sort_index(ascending=False)

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def get_cost(self, symbol, **kwargs):
        """FIXME do not work well sometimes giving a bad result by unknown reason.

        Get weighted average (from buy trades data) cost for a symbol.

        >>> api = PandaXT('binance')
        >>> symbol_cost = api.get_cost('AGI/BTC')
        >>> isinstance(symbol_cost, float)
        True

        :param symbol: a valid exchange symbol.
        :type symbol: Text or Symbol
        :param kwargs: accepted keys are: balance (Balance) and trades (pd.DataFrame)
        :return float: cost calculation result as float type.
        """
        symbol = self.altname(symbol)
        base, quote = symbol.parts if isinstance(symbol, Symbol) else Currency(symbol) + 'BTC'

        # reuse provided balance and trades data (for rate limit save)
        if 'balance' in kwargs:
            balance = kwargs.get('balance')  # type: Wallet
            balance = balance[base] if isinstance(balance, Wallet) else balance
        else:
            balance = self.get_balance(base, field='total')  # type: Balance

        balance = balance.total if balance and hasattr(balance, 'total') else balance

        if balance > 0.0:
            # balance = balance.total
            trades = kwargs.get('trades', self.get_user_trades(symbol))

            buys = trades.query('side == "buy"')

            # if "order" column is None replace it with "id" column
            if buys['order'].isna().all():
                buys['order'].update(buys['id'])

            # groupby operations per column
            columns_op = dict(amount='sum', price='mean', cost='mean', timestamp='mean')
            buys = buys.groupby('order').agg(columns_op).sort_index(ascending=False)  # type: pd.DataFrame
            buys = buys[['price', 'amount']].reset_index(drop=True)

            for index, price, amount in buys.itertuples():
                if balance - amount <= 0:
                    if round(balance - amount, 8) != 0.0:
                        prices, amounts = buys[:index + 1].T.values
                        amounts[-1] = balance
                    else:
                        prices, amounts = buys[:index + 1].T.values
                    return round(pd.np.average(prices, weights=amounts), 10)
                else:
                    balance -= amount

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def get_order_status(self, order_id, symbol=None):
        """Get order status by order_id.

        :param Text order_id: a valid order id.
        :param symbol: a valid exchange market
        :type symbol: str or Symbol or Market
        :return Text: order status as str. Possible values are: "closed",  "canceled", "open"
        """
        return self._api.fetch_order_status(order_id, symbol=symbol)

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def get_open_orders(self, symbol=None):
        """Get open orders.md for a symbol.

        :param symbol: symbol used in opened orders.md server query.
        :type symbol: str or Symbol
        :return List: list of open orders.md for specific symbol.
        """
        assert isinstance(symbol or 0, str) and symbol in self.symbols
        raw = self._api.fetch_open_orders(symbol)

        return [{Symbol(k): v for k, v in r.items() if k not in 'info'} for r in raw.copy()]

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def get_profit(self, currency):
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

    @retry(exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def cancel_order(self, symbol, last_only=False):
        """
        Cancel symbols open orders.md for a symbol.

        :param symbol: the symbol with open orders.md.
        :type symbol: Bool or Symbol
        :param Bool last_only: if True, only last order sent will be cancelled.
        :return List: list of dict with data about cancellations.
        """
        symbol = Symbol(symbol)
        pending_orders = self.get_open_orders(symbol)

        if len(pending_orders):
            if last_only:
                return self._api.cancel_order(pending_orders[-1]['id'], symbol)

            else:
                canceled_orders = list()
                for p in pending_orders:
                    result = self._api.cancel_order(p['id'], symbol)

                    if result and result.get('status', '') in 'cancel':
                        canceled_orders.append({k: v for k, v in result.items() if v})
                    else:
                        self._api.cancel_order(p['id'], symbol)
                return canceled_orders

    def __str__(self):
        """PandaXT instance as "str" type representation.

        >>> str(PandaXT('binance'))
        'binance'

        :return Text: PandaXT instance as "str" type representation.
        """
        return self.id

    def __repr__(self):
        """
        PandaXT instance as "str" type representation.

        >>> PandaXT('binance')
        binance

        :return str: PandaXT instance as "str" type representation.
        """
        return self.id

    def __contains__(self, item):
        """
        Check if a symbol or currency is supported by exchange.

        >>> exchange = PandaXT('cryptopia')
        >>> Currency('ETH') in exchange
        True
        >>> Currency('ETH/BTC') in exchange
        True
        >>> Currency('MyCOIN') in exchange
        False

        :param item: currency or symbol for supports checking on exchange.
        :type item: str or Currency or Symbol
        :return bool: True is item is supported, otherwise False.
        """
        return str(item) in self.markets.keys() or str(item) in map(str, self.currencies)


class Exchanges(UserDict):
    """Dick like class to store multiple exchanges."""

    def __init__(self, **exchanges):
        """Constructor.

        :param list exchanges: exchange names as list
        """
        super().__init__(**exchanges)
        exchanges = self._get_exchange_latest_versions(exchanges)
        self.data.update({v: PandaXT(v) for v in exchanges})

    def __getattr__(self, item):
        item = self._get_latest_version_name(item)
        if item in self._exchanges:
            return PandaXT(self._exchanges.get(item))
        else:
            raise AttributeError('{} is not a supported exchange.'.format(item.upper()))

    def __getitem__(self, item):
        return getattr(self, item)


if __name__ == '__main__':
    a = PandaXT('binance')
    result = a.get_cost('AGI/BTC')
