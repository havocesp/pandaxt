# -*- coding:utf-8 -*-
"""
Utils module.
"""
import os
import pathlib
# from collections import Iterable


def dict_none_drop(d):
    """
    Drop None values from dict "d"

    This function will also round float type values to 8 precision.

    :param dict d: dict to transform.
    :return dict: "d" dict without None values and with all float types rounded to 8 precision.
    """
    result = dict()
    for k, v in d.items():
        if v:
            if isinstance(v, dict):
                v = dict_none_drop(v)

            result.update({k: v})
    return {k: round(v, 8) if isinstance(v, float) else v for k, v in result.items()}


def ctype(v, t):
    """
    Convert "v" to type "t"

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
    r = ctype(v, float) if isinstance(r, float) and not r.is_integer() else ctype(v, int)
    return r if isinstance(r, (float, int)) else v

#
# def numfmt(n, fmt=None):
#     """
#     Get str type from "n" float after apply "fmt" format spec.
#
#     :param float n: float type to be formatted as str.
#     :param str fmt: a str type with a valid float format spec.
#     :return str: "n" float as str type formatted applying "fmt" specs.
#     """
#     if str(n).replace(' ', '').replace('+', '').replace('-', '').isnumeric():
#         if fmt:
#             fmt = fmt.split('.')[0] + fmt.split('f')[1]
#         else:
#             fmt = '{:d}'
#         return fmt.format(int(n))
#     elif isinstance(magic2num(n), float):
#         if fmt is None or not isinstance(fmt, (int, str)):
#             fmt = 8
#             for p, limit in zip([4, 3, 2, 1], [0.1, 1.0, 100.0, 1000.0]):
#                 if magic2num(n) > limit:
#                     fmt = p
#
#         fmt = '{{:.{:d}f}}'.format(fmt) if isinstance(fmt, int) else fmt
#
#         return fmt.format(magic2num(n))
#     else:
#         return n


# def to_array(d):
#     """
#     Return a list type extracted from 'd' param.
#
#     Depending on d param type, the following rules will be applied:
#
#     - If "d" is str type, return [d]
#     - If "d" is dict type, return list(d.values())
#     - If "d" is other iterable type, return [v for v in d]
#     - Otherwise return [d] (or None)
#
#     :param tp.Any d:
#     :return list: a list of dict type extracted from "d" param.
#     """
#     if d is not None:
#         if isinstance(d, Iterable):
#             if isinstance(d, str):
#                 d = [d]
#             elif isinstance(d, dict):
#                 d = list(d.values())
#             else:
#                 d = [v for v in d]
#         else:
#             d = [d]
#     return d
#
#
# def group_by(array, key):
#     """
#     Group array by key.
#
#     :param list array: list of dict types.
#     :param str key: key used to group.
#     :return dict: array grouped  by key.
#     """
#
#     result = dict()
#     array = to_array(array)
#     array = [entry for entry in array if (key in entry) and (entry[key] is not None)]
#     for entry in array:
#         if entry[key] not in result:
#             result[entry[key]] = list()
#         result[entry[key]].append(entry)
#     return result
#
#
# def sort_by(array, key, descending=False):
#     """
#     Sort an "array" by "key" values.
#
#     :param list array: list of dict type.
#     :param str key: key used to compare.
#     :param bool descending: a reverse switch.
#     :return list: a key values sorted array (ascending or descending depending on descending value).
#     """
#     return list(sorted(array, key=lambda k: k[key] if k[key] is not None else "", reverse=descending))
#
#
# def filter_by(array, key, value=None):
#     """
#
#     :param list array:
#     :param str key:
#     :param value:
#     :return list:
#     """
#     if value:
#         grouped = group_by(array, key)
#         if value in grouped:
#             return grouped[value]
#         return list()
#     return array


# def key_values(array, key, skey=None):
#     result = [
#         element[key][skey] if skey and skey in element[key] else element[key]
#         for element in array
#         if (key in element) and (element[key] is not None)
#     ]
#     return result
#
#
# def drop_keys(d, *keys):
#     """
#     Return "d" dict after drop all keys except specified as "keys" var param.
#
#     :param dict d: dict to transform.
#     :param list keys: "d" dict keys to keep.
#     :return dict: "d" dict after drop all keys except specified as "keys" var param.
#     """
#
#     return [{x: y for x, y in r.items() if x not in keys} for r in d.values()]


# def subdict(d, keys, drop_none):
#     """
#
#     :param dict d:
#     :param keys:
#     :type keys: list or set or tuple
#     :param bool drop_none:
#     :return dict:
#     """
#     return {k: v for k, v in d.items() if bool(not drop_none or drop_none and v is not None) and k in keys}


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
