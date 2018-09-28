# -*- coding:utf-8 -*-
"""
Model module.
"""
import collections as col

from pandaxt.core import PandaXT as Exchange


class Range:
    """
    Range class.
    """

    def __init__(self, **kwargs):
        self.max = kwargs.get('max', 0.0) or 0.0
        self.min = kwargs.get('min', 0.0) or 0.0

    def __repr__(self):
        return '(max={max}, min={min})'.format(**vars(self))

    def __str__(self):
        return '(max={max}, min={min})'.format(**vars(self))


class Limits:
    """
    Limits class.
    """
    _template = 'amount={amount}, price={price}, cost={cost}'

    def __init__(self, **kwargs):
        """
        Constructor.

        :param kwargs:
        """
        default = dict(max=0.0, min=0.0)
        self.amount = Range(**kwargs.get('amount', default)) or default
        self.price = Range(**kwargs.get('price', default)) or default
        self.cost = Range(**kwargs.get('cost', default)) or default

    def __repr__(self):
        return self._template.format(**vars(self))

    def __str__(self):
        return self.__repr__()


class Currency(str):
    """
    Currency class.
    """
    _name = None
    _precision = None

    def __new__(cls, name):
        return str.__new__(cls, name.upper())

    def __init__(self, name, precision=8, long_name=None):
        """
        Constructor.

        :param str name:
        :param int precision:
        :param str long_name:
        """
        super(str, Currency).__init__(name)

        # initial_value = str(name if name and len(name) else '')
        self.precision = precision
        self._long_name = long_name

    @property
    def precision(self):
        """

        :return:
        """
        return self._precision

    @precision.setter
    def precision(self, v):
        self._precision = int(v if v and isinstance(v, (int, float)) else 8)

    @property
    def long_name(self):
        return self._long_name or self

    @long_name.setter
    def long_name(self, v):
        if v and isinstance(v, str):
            self._long_name = v
        else:
            self._name = str()

    def __repr__(self):
        return str(self)

    def __mod__(self, other):
        if isinstance(other, Exchange) and other.symbols and len(other.symbols):
            return self in map(str, other.currencies)
        return False

    def __add__(self, other):
        if str(other or str()).upper().rstrip('T') in ['USD', 'EUR', 'BTC']:
            return Symbol('{}/{}'.format(self, other))
        else:
            return None

    def __contains__(self, item):
        if isinstance(item, Exchange) and item.currencies and len(item.currencies):
            return self in item.currencies
        else:
            return item in str(self)


class Symbol(str):
    """
    Symbol class.
    """

    def __new__(cls, symbol=None):
        symbol = str(symbol).upper()
        symbol = symbol if '/' in symbol else '{}/BTC'.format(symbol)
        return str.__new__(cls, symbol)

    @property
    def base(self):
        """
        Base currency.
        :return:
        """
        return Currency(self.split('/')[0])

    @property
    def quote(self):
        """
        Quote currency.
        :return:
        """
        return Currency(self.split('/')[1])

    @property
    def currencies(self):
        sp = self.split('/')
        if len(sp[1:]):
            return Currency(sp[0]), Currency(sp[1])
        else:
            return Currency(sp[0]), Currency('BTC')

    def __str__(self):
        return '{}/{}'.format(*self.currencies)

    def split(self, sep=None, maxsplit=-1):
        return str.split(self, sep=str(sep or '/'))

    def __contains__(self, item):
        if isinstance(item, Exchange) and item.symbols and len(item.symbols):
            return self in item.symbols
        else:
            return item in str(self)

    @property
    def parts(self):
        return self.base, self.quote


# class Symbol(str):
#     def __init__(self, symbol=None):
#         self.id = str(symbol or str()).upper()
#         if len(self.id):
#             assert '/' in symbol, 'Invalid symbol: {}'.format(symbol)
#         super().__init__(o=symbol)
#
#     @property
#     def base(self):
#         """
#         Base currency.
#         :return:
#         """
#         return Currency(self.split('/')[0]) if '/' in self else Currency()
#
#     @property
#     def quote(self):
#         return Currency(self.split('/')[1]) if '/' in self else Currency()
#


class Precision:
    """
    Market precision class.
    """
    _template = 'amount={amount}, price={price}'

    def __init__(self, **kwargs):
        """
        Constructor.

        :param kwargs: accepted keys: amount, price
        """
        self.amount = kwargs.get('amount', 0) or 0
        self.price = kwargs.get('price', 0) or 0

    def __repr__(self):
        return self._template.format(**vars(self))

    def __str__(self):
        return self.__repr__()


class Market(col.OrderedDict):
    """
    Market class.
    """

    def __init__(self, **kwargs):
        """
        Constructor.

        :param kwargs: accepted keys: precision, limits, precision, id, symbol, percentage, info, base, quote, baseId, quoteId, numericId, label, maker, taker, active
        """
        super().__init__(**kwargs)
        self.precision = Precision(**kwargs.get('precision', dict()))
        self.limits = Limits(**kwargs.get('limits', dict()))
        self.precision = Precision(**kwargs.get('precision', dict()))
        self.id = kwargs.get('id', str())
        self.symbol = Symbol(kwargs.get('symbol', str()))
        self.percentage = kwargs.get('percentage')
        self.base = Currency(kwargs.get('base', str()))
        self.quote = Currency(kwargs.get('quote', str()))
        self.baseId = kwargs.get('baseId', str())
        self.quoteId = kwargs.get('quoteId', str())
        self.active = kwargs.get('active', False)
        # self.info = kwargs.get('info', dict())
        self.numericId = kwargs.get('numericId')
        self.maker = kwargs.get('maker', 0.0) or 0.0
        self.taker = kwargs.get('taker', 0.0) or 0.0
        self.label = kwargs.get('label', str())

    def __str__(self):
        return '\n'.join([' - {:<16} = {:>32}'.format(k, str(v) or str()) for k, v in vars(self).items()])

    def __repr__(self):
        return '\n'.join([' - {:<16} = {:>32}'.format(k, str(v) or str()) for k, v in vars(self).items()])


if __name__ == '__main__':
    # markets = Exchange('binance').markets
    s = Symbol('XRP/BTC')
    # m = markets.get()

    # dict2class(m['precision'])
