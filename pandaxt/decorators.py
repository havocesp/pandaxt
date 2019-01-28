# -*- coding: utf-8 -*-
"""Decorators module
 
 - Package:     pandaxt
 - Version:     $Version
 - Author:      Daniel J. Umpierrez
 - Created:     06-10-2018
 - License:     MIT
 - Site:        https://github.com/havocesp/pandaxt
"""
import time
from functools import wraps

import requests

__all__ = ['extract_data', 'response_error_raise', 'get']


def extract_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        data = response['Data']
        return data

    return wrapper


def retry(**params):

    def decorator(func):

        def wrapper(*args, **kwargs):
            exceptions = params.get('exceptions', Exception)
            retries = params.get('retries', 3)
            interval = params.get('interval', 5)
            retries = retries if retries and isinstance(retries, int) and retries > 0 else 3
            interval = interval if interval and isinstance(interval, int) and interval >= 1 else 1
            data = None
            while retries > 0 and data is None:
                try:
                    data = func(*args, **kwargs)
                    return data.get('Data', data) if isinstance(data, dict) else data
                except exceptions:
                    retries -= 1
                    time.sleep(interval)
            return data

        return wrapper

    return decorator


def response_error_raise(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        resp = func(*args, **kwargs)  # type: requests.Response
        if resp is None:
            raise ConnectionError('Data received from server is not valid or null')
        elif not resp.ok:
            resp.raise_for_status()
        return resp

    return wrapper


def get(url):
    def decorator(func):
        def wrapper(*args, **kwargs):
            params = func(*args, **kwargs)
            response = requests.get(url, params=params)
            return response.json()

        return wrapper

    return decorator
