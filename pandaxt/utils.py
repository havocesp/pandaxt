# -*- coding:utf-8 -*-
"""
Utils module.
"""
import os
import pathlib


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


def sort_by(array, key, descending=False):
    """
    Sort an "array" by "key".

    :param list array:
    :param str key:
    :param bool descending:
    :return :
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
