# PandaXT

A Python 3 "__ccxt__" wrapper over __Pandas__ lib.

 - Author: Daniel J. Umpierrez
 - License: UNLICENSE
 - Version: 0.1.10

## Description

Python ccxt multi-exchange lib mixed with pandas lib for data handling.

## Requirements

 - [requests](https://pypi.org/project/requests)
 - [regex](https://pypi.org/project/regex)
 - [ccxt](https://pypi.org/project/ccxt)
 - [pandas](https://pypi.org/project/pandas)
 - [Cython](https://pypi.org/project/Cython)
 - [diskcache](https://pypi.org/project/diskcache)
 - [newtulipy](https://pypi.org/project/newtulipy/)

## Usage

### Basic example

```python
from pandaxt import PandaXT
api = PandaXT('binance')
print(api.markets)
```

## Changelog

Project changes over versions.

### 0.1.10
 - Update python versions support ti >=3.8
 - Update some requirements.
 - Added tox.ini and pyproject.toml files.

### 0.1.9
 - Removed unnecessary comments and code at utils.py
 - Added docstring to to_lower function.

### 0.1.8
 - Minor fixes.

### 0.1.7
 - Fixed "currencies" method and recursion limit bugs.

### 0.1.6
- README requirements fixed
- Removed 'PandaXT._fetch_ohlcv' method.

### 0.1.5
- Many features added and many errors fixed

### 0.1.4
- New "model" module.
- Added dict2class in "utils" module.

### 0.1.3
- "Binance" specific options added for time adjust and auto fit precision or orders.md.

### 0.1.2
- New "create_market_order" method.
- New "sort_by" function in utils module.
- Unified cancel methods into "cancel_order".
- Sell and Buy methods auto-fill price and amount if not is supplied as param.

### 0.1.1
- Added precision related methods to "_PandaXT_" class.

### 0.1.0
- Initial version

## TODO
- TODO
