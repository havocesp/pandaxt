# -*- coding:utf-8 -*-
"""Utils module."""
import collections as col
import inspect
import os
import pathlib
import typing as tp
from itertools import repeat, starmap

import pandas as pd
import requests


def error(text=None):
    print(' - [ERROR] {}'.format(text))


def repeatfunc(func, times=None, *args):
    """Repeat calls to func with specified arguments.

    Example:  repeatfunc(random.random)
    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))


def get_timeframe(d, tf):
    """Get most close value from array.

    >>> tf = {'1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '2h': '2h', '4h': '4h', \
    '6h': '6h', '8h': '8h', '12h': '12h', '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'}

    >>> get_timeframe(tf.values(), '10m')
    '15m'

    :param tp.Dict d: array with data where closest value will be search
    :param str tf: value to find.
    """
    unit = d[-1:]
    value = int(d[:-1])
    d = [e[-1] for e in d.values() if str().endswith(unit) == unit]

    by_unit = pd.np.asarray(d)
    idx = (pd.np.abs(by_unit - value)).argmin()
    return [idx]


def whoami():
    """Return function name where this function is called."""
    frame = inspect.currentframe()
    return inspect.getframeinfo(frame).function


def drop_keys(it, *keys):
    """
    Remove specified "keys" from any iterable type (tuple, list, dict, ...).

    :param tp.Iterable it: iterable containing dict type instances (or similar)
    :param tp.List[tp.AnyStr] keys: keys to be dropped.
    :return tp:Iterable: iterable "it" after keys deletion.
    """
    if it is not None:
        if isinstance(it, col.Mapping) and len(it):
            it = dict(it)
            if 'info' in it:
                del it['info']
                return it
            else:
                return {k: drop_keys(v) for k, v in it.items() if k not in keys}
        elif isinstance(it, tp.Iterable):
            return [drop_keys(v) if isinstance(v, col.Mapping) else v for v in it]
    else:
        return it


def flt(value, p=None, as_str=False):
    """
    Float builtin wrapper for precision param initialization.

    >>> flt(4.13, 5, False)
    4.13
    >>> flt(4.13, 5, True)
    '4.13'
    >>> flt(4, 5, True)
    '4'

    :param bool as_str: return as string
    :param value:
    :type value: Number or str
    :param int p:
    :return:
    :rtype: str or Number
    """
    value_str = str(value or '0.0').strip()

    if value_str.strip().lstrip('+-').replace('.', '').isnumeric():
        template = '{{:.{:d}f}}'.format(p or infer_precision(value))
        value = template.format(float(value_str)).rstrip('.0')

        try:
            if not as_str:
                return int(value) if float(value).is_integer() else float(value)
            else:
                return value
        except ValueError:
            return value
    else:
        return value


def infer_precision(number):
    """
    Infer precision base on number size.

    >>> infer_precision(0.343)
    3
    >>> infer_precision(0.0001343)
    5
    >>> infer_precision(12300)
    0

    :param float number: number used to infer precision.
    :return int: precision as int (number of decimals recommended)
    """
    number = number or 0.0
    try:
        number = float(number)
    except ValueError:
        return number
    if number is not None and isinstance(number, float):
        if number < 0.000001:
            return 10
        elif number < 0.0001:
            return 8
        elif number < 0.01:
            return 5
        elif number < 1.0:
            return 3
        elif number < 100.0:
            return 2
        elif number < 1000.0:
            return 1
        else:
            return 0


def num2str(n, precision=None):
    """
    Numeric type infer and parser

    Accept any Iterable (dict, list, tuple, set, ...) or built-in data types int, float, str, ... and try  to
    convert it a number data type (int, float)

    >>> num2str(['10.032', '10.32032', '11.392', '13'])
    [10.03, 10.32, 11.39, 13]

    :param n: number
    :type n: Number or tp.Iterable
    :param int precision:
    :return tp.Iterable:
    """
    if n is not None:
        if isinstance(n, str):
            backup = type(n)(n)
            try:
                precision = precision if isinstance(precision or '', int) else infer_precision(n)
                n = flt(n, precision)
            except ValueError:
                n = backup
        if isinstance(n, float):
            precision = precision if isinstance(precision or '', int) else infer_precision(n)
            n = int(n) if n.is_integer() else round(n, precision)
        elif isinstance(n, str):
            n = flt(n, precision, as_str=True)
        elif isinstance(n, tp.Mapping):
            n = {k: num2str(v, precision) if isinstance(v, col.Iterable) else v for k, v in dict(n).items()}
        elif isinstance(n, tp.Iterable):
            n = [num2str(n, precision) if isinstance(n, col.Iterable) else n for n in list(n)]
    return n


def dict_none_drop(d, default=None):
    """
    Drop None values from dict "d"

    This function will also round float type values to 8 precision.

    >>> dict_none_drop({'data': {'a': 10, 'b': None, 'c': True}})
    {'data': {'a': 10, 'c': True}}

    >>> dict_none_drop({'data': {'a': 10, 'b': None, 'c': [0.2, None, '100']}}, 0.0)
    {'data': {'a': 10, 'b': 0.0, 'c': [0.2, 0.0, '100']}}

    >>> dict_none_drop({'data': {'a': 10, 'b': None, 'c': [0.2, None, '100']}})
    {'data': {'a': 10, 'c': [0.2, '100']}}

    :param dict d: dict to transform.
    :param tp.Any default: None replacement value (default None)
    :return dict: "d" dict without None values and with all float types rounded to 8 precision.
    """
    result = dict()
    for k, v in d.items():
        if v is not None:
            if isinstance(v, dict):
                v = dict_none_drop(v, default)
            elif not isinstance(v, str) and isinstance(v, tp.Iterable):
                v = [e or default for e in list(v) if e is not None or default is not None]
        elif default is not None:
            v = default
        if v is not None:
            result.update({k: v})

    return {k: round(v, 8) if isinstance(v, float) else v for k, v in result.items()}


def ctype(v, t):
    """
    Convert "v" to type "t"

    >>> ctype('10', int)
    10
    >>> ctype('10', list)
    ['1', '0']
    >>> ctype(10, list)
    10
    >>> ctype('10a', float)
    '10a'

    :param tp.Any v:
    :param tp.TypeVar t:
    :return tp.Any:
    """
    try:
        return t(v) or v
    except (ValueError, TypeError):
        return v


def magic2num(v):
    """
    Try to parse "v" to a built-in numeric type (float or int), otherwise "v" will be returned unchanged.

    :param str v: str to parse as numeric type.
    :return: a parsed "v" as built-in numeric type (float or int), otherwise "v" will be returned unchanged.
    :rtype: tp.Any
    """
    r = ctype(v, float)
    r = r if isinstance(r, float) and not r.is_integer() else ctype(v, int)
    return r if isinstance(r, (float, int)) else v


# noinspection PyUnusedFunction
def sort_by(array, key, descending=False):
    """
    Sort an "array" by "key".

    >>> sort_by([{'a': 10, 'b': 0}, {'a': 5, 'b': -1}, {'a': 15, 'b': 1}], 'a')
    [{'a': 5, 'b': -1}, {'a': 10, 'b': 0}, {'a': 15, 'b': 1}]

    :param tp.List[tp.Mapping] array: list of dicts.
    :param str key: key name to use when sorting.
    :param bool descending: if True sort will be descending.
    :return tp.List[tp.Mapping]: a by "key" sorted list.
    """
    return list(sorted(array, key=lambda k: k[key] if k[key] is not None else "", reverse=descending))


def load_dotenv(env_path=None):
    """
    Load ".env" file to environment (accessible through "os.environ").

    :param str env_path: str containing path to ".env" file (default "$HOME/.env")
    """

    def env2dict(s):
        if len(s) and '=' in s and s[0].isalpha():
            k, v = s.split('=', maxsplit=1)
            return {k: str(v).strip('"\'')}

    env_file = pathlib.Path.home().joinpath(env_path or '.env')

    if env_file.exists():
        content = env_file.read_text()
        lines = content.split('\n')
        for ln in [l for l in lines if len(l or '')]:
            os.environ.update(env2dict(ln))


# noinspection PyUnusedFunction
def dict2class(d, class_name):
    """
    Convert a dict instance to python code as a str class.
    :param d:
    :param class_name:
    :return:
    """

    template = """class {}:
    def __init__(self, *args, **kwargs):
        pass
    {}
    """
    print(template.format(class_name, ''.join(['    {} = {}()\n'.format(k, type(v).__name__) for k, v in d.items()])))


def get_tor_session(port=9050):
    """
    Get a "Session" instance to socks5 tor proxy.

    :param int port: the tor service port (default 9050)
    :return requests.Session: a requests Session like instance ready for tor network connection.
    """
    session = requests.session()
    # Tor uses the 9050 port as the default socks port
    session.proxies = {
        "http": "socks5://127.0.0.1:{:d}".format(port),
        "https": "socks5://127.0.0.1:{:d}".format(port)
    }
    return session
