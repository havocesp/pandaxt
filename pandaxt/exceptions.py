# -*- coding:utf-8 -*-
"""exceptions module"""


class TimeframeError(ValueError):

    def __init__(self, timeframe, exchange=None):
        exchange = f' {str(exchange)}' if exchange else '.'
        super().__init__(f'{str(timeframe)} is not a valid timeframe for exchange {exchange}')


class SideError(ValueError):

    def __init__(self, side):
        super().__init__(f'Invalid side: {side} (accepted values: "buy", "sell")')


class SymbolError(ValueError):

    def __init__(self, symbol=None, exchange=None):
        exchange = f' {exchange}' if exchange else str()
        symbol = f' {str(symbol)}' if symbol else str()
        super().__init__(f'{symbol} is not listed at exchange {exchange}')


class CurrencyError(ValueError):

    def __init__(self, currency=None, exchange=None):
        exchange = f' {exchange}' if exchange else str()
        currency = f' {str(currency)}' if currency else str()
        super().__init__(f'{currency} is not listed at exchange {exchange}')
