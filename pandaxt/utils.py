# -*- coding:utf-8 -*-
"""Utils module."""
import inspect
import os
import pathlib
from itertools import repeat, starmap
from typing import Any, Callable, Iterable as Iter, Mapping as Map, NoReturn as Void, Text as Str

Int = int
Float = float
Bool = bool


def error(text=None) -> Void:
    """Print a formatted error text.

    :param Str text: error message.
    """
    print(f' - [ERROR] {text}')


def repeatfunc(func: Callable, times: Int = None, *args):
    """Repeat calls to func with specified arguments.

    Example:  repeatfunc(random.random)

    :param func: function to be called
    :param times: amount of call times
    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))


def whoami() -> Str:
    """Return function name where this function is called.

    :return: caller function name.
    """
    frame = inspect.currentframe()
    return inspect.getframeinfo(frame).function


def drop_keys(it, *keys):
    """Remove specified "keys" from any iterable type (tuple, list, dict, ...).

    :param tp.Iterable it: iterable containing dict type instances (or similar)
    :param tp.List[tp.AnyStr] keys: keys to be dropped.
    :return tp:Iterable: iterable "it" after keys deletion.
    """
    if it is not None:
        if isinstance(it, Map) and len(it):
            it = dict(it)
            if 'info' in it:
                del it['info']
                return it
            else:
                return {k: drop_keys(v)
                        for k, v in it.items() if k not in keys}
        elif isinstance(it, Iter):
            return [drop_keys(v) if isinstance(v, Map) else v for v in it]
    else:
        return it


def flt(value, p=None, as_str=False):
    """Float builtin wrapper for precision param initialization.

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
        template = '{{:.{:d}f}}'.format(p or auto_precision(value))
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


def to_lower(obj: Any) -> Any:
    """Lower case conversion of practically any str convertible type.

    :param obj: str convertible type to be "lower cased"
    :return: obj as str lower cased.
    """
    if obj:
        tt = type(obj)
        if isinstance(obj, Str):
            obj = obj.lower()
        elif isinstance(obj, Map):
            obj = tt({k.lower(): v for k, v in obj.items()})
        elif isinstance(obj, Iter):
            obj = tt([v.lower() if isinstance(v, Str) else v for v in obj])
    return obj


def apply(obj: Any, func: Callable) -> Any:
    """Run a function using as argument "obj".

    If "obj" is an iterable or mapping, the function is executed as many times
    as the number of elements of "obj".

    :param obj: object where the func will by applied.
    :param func: function to apply to obj or obj elements.
    :return: "obj" after the function application is done.
    """
    if obj:
        tt = type(obj)
        if isinstance(obj, Map):
            obj = tt({k.lower(): func(v) for k, v in obj.items()})
        elif isinstance(obj, Iter):
            obj = tt([func(v) for v in obj])
        else:
            obj = func(obj)
    return obj


def auto_precision(num) -> float:
    """Infer precision base on number size.

    >>> auto_precision(0.34388)
    0.3439
    >>> auto_precision(0.0001343)
    0.0001343
    >>> auto_precision(12300)
    12300

    :param float num: number used to infer precision.
    :return: precision as int (number of decimals recommended)
    """
    try:
        num = float(num)
    except ValueError:
        return num

    if isinstance(num, float):
        for v in [(e, 10000.0 / (10 ** e)) for e in range(8, 0, -1)]:
            precision, cutoff = v
            if num < cutoff:
                return round(num, precision)
        else:
            return round(num)


# def num2str(value, precision=None):
#     """Numeric type infer and parser.
#
#     Accept any Iterable (dict, list, tuple, set, ...) or built-in data types int, float, str, ... and try  to
#     convert it a number data type (int, float)
#
#     >>> num2str(['10.032', '10.32032', '11.392', '13'])
#     [10.03, 10.32, 11.39, 13]
#
#     :param value: number
#     :type value: float or int or Iterable
#     :param int precision:
#     :return tp.Iterable:
#     """
#     if value is not None:
#         if isinstance(value, str):
#             backup = type(value)(value)
#             try:
#                 if isinstance(precision, (str, float)):
#                     precision = auto_precision(value)
#                 value = flt(value, precision)
#             except ValueError:
#                 value = backup
#         if isinstance(value, float):
#             precision = precision if isinstance(
#                 precision or '', int
#             ) else auto_precision(value)
#             value = int(value) if value.is_integer() else round(value, precision)
#         elif isinstance(value, str):
#             value = flt(value, precision, as_str=True)
#         elif isinstance(value, Map):
#             value = {k: num2str(v, precision) if isinstance(v, Iter) else v for k, v in dict(value).items()}
#         elif isinstance(value, Iter):
#             value = [num2str(n, precision) if isinstance(n, Iter) else n for n in list(value)]
#     return value


def dict_none_drop(d, default=None):
    """Drop None values from dict "d"

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
    result = {}
    for k, v in d.items():
        if v is not None:
            if isinstance(v, dict):
                v = dict_none_drop(v, default)
            elif not isinstance(v, str) and isinstance(v, Iter):
                v = [e or default for e in list(v) if e is not None or default is not None]
        elif default is not None:
            v = default
        if v is not None:
            result.update({k: v})

    return {k: round(v, 8) if isinstance(v, float) else v for k, v in result.items()}


def ctype(v, t):
    """Convert "v" to type "t"

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
    """Try to parse "v" to a built-in numeric type (float or int), otherwise "v" will be returned unchanged.

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
    """Load $HOME/.env file to environment (accessible through "os.environ").

    :param str env_path: str containing path to ".env" file (default "$HOME/.env")
    """
    env_file = pathlib.Path.home().joinpath(env_path or '.env')

    if env_file.is_file():
        # print(' - [pandaxt.utils][DEBUG] Reading ".env" file ...')
        content = env_file.read_text()
        lines = content.split('\n')
        for ln in [ll for ll in lines if len(ll or '')]:
            if '=' in ln and ln[0].isupper():
                k, v = ln.split('=', maxsplit=1)
                if k and v and len(k) and len(v):
                    os.environ.update({k: v.strip("\"'")})
