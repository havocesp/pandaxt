# PandaXT

A Python 3 "__ccxt__" wrapper over __Pandas__ lib.

 - Author: Daniel J. Umpierrez
 - License: MIT
 - Version: 0.1.4

## Description

Python ccxt multi-exchange lib mixed with pandas lib for data handling.

## Requirements

 - [ccxt](https://github.com/ccxt/ccxt)
 - [pandas](https://github.com/pandas-dev/pandas)
 - [tulipy](https://github.com/cirla/tulipy)

## Usage

### Basic example

```python
from pandaxt import PandaXT
api = PandaXT('binance')
print(api.markets)

```

## Changelog

Project changes over versions.

### 0.1.4
- New "model" module.
- Added dict2class in "utils" module.

### 0.1.3
- "Binance" specific options added for time adjust and auto fit precision or orders.

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