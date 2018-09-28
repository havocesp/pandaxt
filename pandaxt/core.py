# -*- coding:utf-8 -*-
"""
Core module.
"""
import os
from collections import OrderedDict

import ccxt.binance
import pandas as pd
import tulipy

from pandaxt.model import Currency
from pandaxt.utils import load_dotenv, dict_none_drop, magic2num

OHLC_FIELDS = ['date', 'open', 'high', 'low', 'close', 'volume']

SETTINGS = dict(config=dict(timeout=25000, enableRateLimit=True))

__all__ = ['PandaXT']


class PandaXT:
    """
    "ccxt" exchanges wrapper class over Pandas lib.
    """
    _api = None  # type: ccxt.binance

    def __init__(self, exchange, load_markets=True, load_keys=True):
        """
        Constructor.

        :param str exchange: a ccxt lib supported exchange
        :param bool load_markets: if True, "load_markets" method will be executed.
        :param bool load_keys:  if True, exchange API keys will be load from "$HOME/.env" file.
        """
        assert str(exchange).lower() in ccxt.exchanges, '{} not supported'.format(str(exchange))
        api = getattr(ccxt, str(exchange).lower())
        settings = SETTINGS.get('config')

        if load_keys:
            load_dotenv()
            self.key = os.environ.get('{}_KEY'.format(exchange.upper()))
            self.secret = os.environ.get('{}_SECRET'.format(exchange.upper()))

            if self.key and self.secret:
                settings.update(apiKey=self.key, secret=self.secret)

        self._api = api(config=settings)

        if exchange in 'binance':
            self._api.load_time_difference()
            self._api.options['parseOrderToPrecision'] = True
        elif exchange in 'cryptopia':
            self._api.fetch_ohlcv = ccxt.Exchange().fetch_ohlcv
        if load_markets:
            self._api.load_markets()

    @property
    def name(self):
        # noinspection PyUnresolvedReferences
        return self._api.name

    def cost2precision(self, symbol, cost):
        """
        Return cost rounded to symbol precision exchange specifications.

        :param str symbol: a valid exchange symbol.
        :param float cost: cost to be rounded.
        :return float: cost rounded to specific symbol exchange specifications.
        """
        return float(self._api.cost_to_precision(symbol, cost))

    def amount2precision(self, symbol, amount):
        """
        Return amount rounded to symbol precision exchange specifications.

        :param str symbol: a valid exchange symbol.
        :param float amount: amount to be rounded.
        :return float: amount rounded to specific symbol exchange specifications.
        """
        return float(self._api.amount_to_precision(symbol, amount))

    def price2precision(self, symbol, price):
        """
        Return price rounded to symbol precision exchange specifications.

        :param str symbol: a valid exchange symbol.
        :param price: price to be rounded.
        :type price: int or float or str
        :return float: price rounded to specific symbol exchange specifications.
        """
        return float(self._api.price_to_precision(symbol, price))

    def get_price_precision(self, symbol):
        """
        Get price precision set by exchange for a symbol.

        :param symbol: a valid exchange symbol.
        :return int: price precision set by exchange for "symbol".
        """
        return int(self._api.markets[symbol]['precision']['price'])

    def get_amount_precision(self, symbol):
        """
        Get amount precision set by exchange for a symbol.

        :param symbol: a valid exchange symbol.
        :return int: amount precision set by exchange for "symbol".
        """
        return int(self._api.markets[symbol]['precision']['amount'])

    def get_cost_precision(self, symbol):
        """
        Get cost precision set by exchange for a symbol.

        :param symbol: a valid exchange symbol.
        :return int: cost precision set by exchange for "symbol".
        """
        return int(self._api.markets[symbol]['precision']['cost'])

    @property
    def currencies(self):
        """
        Contains all exchange supported currencies as a alphabetically sorted list.

        :return list: all exchange supported currencies as list type (sorted alphabetically).
        """
        return list(sorted(list(set([Currency(s.split('/')[0]) for s in self.symbols]))))

    @property
    def symbols(self):
        """
        Contains all exchange supported symbols as a alphabetically sorted list.

        :return list: all exchange supported symbols as list type (sorted alphabetically).
        """
        return list(sorted([Symbol(s) for s in self.markets.keys()]))

    @property
    def markets(self):
        """
        Get all exchange markets metadata.

        :return dict: all exchange markets metadata.
        """
        base_markets = ['BTC', 'USD', 'EUR']
        return {k: Market(**{x: y for x, y in v.items() if y and x not in ['info']})
                for k, v in self._api.load_markets().items()
                if k.split('/')[1].rstrip('T') in base_markets}

    def get_ohlc(self, symbol, timeframe='15m', limit=25):
        symbol = str(symbol).upper()
        assert symbol in self._api.load_markets(), '{} not supported'.format(symbol)
        data = self._api.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=OHLC_FIELDS)
        df.index = pd.to_datetime(df.pop('date') // 1000, unit='s')
        df.name = 'date'
        return df

    def get_tickers(self, *symbols):
        """
        Get tickers dict with symbol name as keys for all symbols specified at "symbols" param.

        >>> self.get_ticker("ADA/BTC")
        {"ADA/BTC": {
            'ask': 1.085e-05,
            'askVolume': 5051.0,
            'average': None,
            'baseVolume': 131194124.0,
            'bid': 1.083e-05,
            'bidVolume': 31371.0,
            'change': -8.2e-07,
            'close': 1.084e-05,
            'datetime': '2018-09-11T16:43:03.658Z',
            'high': 1.176e-05,
            'last': 1.084e-05,
            'low': 1.076e-05,
            'open': 1.166e-05,
            'percentage': -7.033,
            'previousClose': 1.167e-05,
            'quoteVolume': 1493.12683047,
            'symbol': 'ADA/BTC',
            'timestamp': 1536684183658,
            'vwap': 1.138e-05
            }
        }

        :param symbols: list of valid exchange symbols.
        :return dict: dict type with tickers data.
        """
        symbols = [Symbol(str(s).upper()) for s in symbols if s in self.symbols]
        assert len(symbols), 'There is some invalid symbol/s in {}'.format(symbols)

        if len(symbols) > 1:
            tickers = self._api.fetch_tickers(symbols)
        else:
            tickers = self._api.fetch_ticker(symbols[0])

        return dict_none_drop(tickers)

    get_ticker = get_tickers

    def get_indicators(self, indicators, symbol, timeframe, limit=25, **indicators_params):
        """
        Get technical analysis indicators data for a symbol.

        :param list indicators: list of valid indicators.
        :param str symbol:  a valid exchange symbol.
        :param str timeframe: a valid timeframe symbol (check exchange official API).
        :param int limit: a valid exchange limit for returned rows (check exchange official API)
        :param dict indicators_params:
        :return dict:
        """
        if isinstance(indicators, str):
            indicators = [indicators]
        indicators = [str(i).lower() for i in indicators]
        symbol = str(symbol).upper()
        result = OrderedDict.fromkeys(indicators)

        functions = OrderedDict.fromkeys(indicators)
        for i in indicators:
            functions.update({i: getattr(tulipy, i)})
        data = self.get_ohlc(symbol, timeframe, limit)
        for n, fn in functions.items():
            inputs = ['close' if i in 'real' else i for i in fn.inputs]
            options = [o.replace(' ', '_') for o in fn.options]
            params = {k: v for k, v in indicators_params.items() if k in options}
            try:
                raw = fn(*data[inputs].T.values, **params)
                di = data.index
                if n in 'roc':
                    raw = raw * 100.0
                sr = pd.Series(raw, name=n.upper())
                sr.index = di.values[-len(sr):]
                result[n] = sr.copy(True)

            except tulipy.lib.InvalidOptionError as err:
                print(str(err))
                print(fn.options)

        return result

    @property
    def id(self):
        """
        Exchange unique reference (also know as ID).

        :return str: exchange unique reference.
        """
        return self._api.id

    @property
    def timeframes(self):
        """
        Return valid exchange timeframes as list.

        :return list: valid exchange timeframes.
        """
        # noinspection PyUnresolvedReferences
        items = self._api.timeframes.items()
        od = OrderedDict(sorted(items, key=lambda x: x[1]))
        return list(od.keys())

    def create_market_order(self, symbol, side):
        """
        Create a market order order.

        :param str symbol: a valid exchange symbol.
        :param str side: accepted values: "buy", "sell"
        :return dict: order creation result data as dict.
        """
        symbol = str(symbol).upper()
        side = str(side).lower()

        assert symbol in self.symbols, 'Invalid symbol: {}'.format(symbol)
        assert side in ['buy', 'sell'], 'Invalid side: {} (accepted values: "buy", "sell")'.format(side)

        base, quote = symbol.split('/')

        if side in 'buy':
            amount = magic2num(self.get_balances('free').get(quote))
        else:
            amount = magic2num(self.get_balances('free').get(base))

        return self._api.create_order(symbol, type='market', side=side, amount=amount)

    def buy(self, symbol, amount=None, price=None):
        """
        Create buy order.

        :param str symbol: a valid exchange symbol.
        :param float amount: amount to buy or None to auto-fill
        :param float price: buy price or None to auto-fill
        :return dict: order creation result data as dict.
        """
        symbol = str(symbol).upper()
        assert symbol in self.symbols, 'Invalid symbol: {}'.format(symbol)
        base, quote = symbol.split('/')
        amount = magic2num(amount or self.get_balances('free').get(quote))
        price = magic2num(price or self.get_ticker(symbol).get('ask'))
        amount = amount / price
        return self._api.create_order(symbol, type='limit', side='buy', amount=amount, price=price)

    def sell(self, symbol, amount=None, price=None):
        """
        Create sell order.

        :param str symbol: a valid exchange symbol.
        :param float amount: amount to sell or None to auto-fill
        :param float price: sell price or None to auto-fill
        :return dict: order creation result data as dict.
        """
        """
        Create buy order.

        :param str symbol:
        :param float amount:
        :param float price:
        :return dict:
        """
        symbol = str(symbol).upper()
        assert symbol in self.symbols, 'Invalid symbol: {}'.format(symbol)
        base, quote = symbol.split('/')
        amount = magic2num(amount or self.get_balances('free').get(base))
        price = magic2num(price or self.get_ticker(symbol).get('bid'))

        return self._api.create_order(symbol, type='limit', side='sell', amount=amount, price=price)

    def get_balances(self, field='total'):
        """
        Get balances.

        :param str field: accepted values: total, used, free
        :return dict : positive balances.
        """
        raw = self._api.fetch_balance()
        data = raw.pop(field)
        return {k: v for k, v in data.items() if v > 0.0}

    def get_balance(self, currency, field='total'):
        """
        Get balance for a currency.

        :param currency: a valid exchange currency.
        :param str field: accepted values: total, used, free
        :return dict: currency with balance amount as float.
        """
        currency = str(currency).upper()
        assert currency in self.currencies, '{} exchange do not support {} currency'.format(self.id.title(), currency)
        balance_data = self.get_balances(field=field) or {currency: 0.0}
        return balance_data.get(currency, {currency: 0.0})

    def get_user_trades(self, symbol, limit=25):
        """
        Get user trades filter by symbol.

        :param symbol: a valid exchange symbol
        :param int limit: a valid limit for rows return (please, refer to official exchange API manual for details)
        :return pd.DataFrame: user trades as pandas DataFrame type.
        """
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

    def get_cost(self, symbol, **kwargs):
        """
        Get weighted average (from buy trades data) cost for a symbol.

        :param str symbol: a valid exchange symbol.
        :param dict kwargs:
        :return float: cost calculation result as float type.
        """
        symbol = str(symbol).upper()

        if '/' not in symbol:
            symbol = '{}/BTC'.format(symbol)
        base, quote = symbol.split('/')

        balance = kwargs.get('balance', self.get_balances().get(base))
        trades = kwargs.get('trades', self.get_user_trades(symbol))

        buys = trades.query('side == "buy"')
        columns_op = {'amount': 'sum', 'price': 'mean', 'cost': 'mean', 'timestamp': 'mean'}
        buys = buys.groupby('order').agg(columns_op).sort_index(ascending=False)  # type: pd.DataFrame

        buys = buys[['price', 'amount']].reset_index(drop=True)
        for index, price, amount in buys.itertuples():
            if balance - amount <= 0:

                if round(balance - amount, 8) != 0.0:
                    prices, amounts = buys[:index + 2].T.values
                    amounts[-1] = balance
                else:
                    prices, amounts = buys[:index + 1].T.values
                return pd.np.average(prices, weights=amounts)
            else:
                balance -= amount

    # no inspection PyUnusedFunction
    def get_order_status(self, order_id, market=None):
        """
        Get order status by order_id.

        :param str order_id: a valid order id.
        :param str market: a valid exchange market
        :return str: order status as str. Possible values are: "closed",  "canceled", "open"
        """
        return self._api.fetch_order_status(order_id, market=market)

    def get_open_orders(self, symbol):
        """
        Get open orders for a symbol.

        :param str symbol: symbol used in opened orders server query.
        :return list: list of open orders for specific symbol.
        """
        raw = self._api.fetch_open_orders(symbol)
        for _ in raw.copy():
            del raw['info']
        return raw

    def cancel_order(self, symbol, last_only=False):
        """
        Cancel symbols open orders for a symbol.

        :param str symbol: the symbol with open orders.
        :param bool last_only: if True, only last order sent will be cancelled.
        :return list: list of dict with data about cancellations.
        """

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
                        self._api.cancel_order(pending_orders['id'], symbol)
                return canceled_orders


if __name__ == '__main__':
    from model import Market, Symbol

    s = Symbol('ETN/BTC')
    # print(s)
    api = PandaXT('cryptopia')
    markets = api.markets
    m = markets.get(s)  # type: Market
    # m = Market(**api.markets[])
    print(m.limits)
    # dict2class(['limits']['cost'], 'cost')
