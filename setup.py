# -*- coding:utf-8 -*-
from setuptools import setup, find_packages

import pandaxt as pxt

requirements = ['requests', 'pandas', 'ccxt', 'tulipy', 'begins', 'py-term']

exclude = ['.idea*', 'build*', '{}.egg-info*'.format(pxt.__package__), 'dist*', 'venv*', 'doc*', 'lab*']

classifiers = [
    'Development Status :: 5 - Production',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]

setup(
    name=pxt.__package__,
    version=pxt.__version__,
    packages=find_packages(exclude=exclude),
    url=pxt.__site__,
    license=pxt.__license__,
    keywords=pxt.__keywords__,
    author=pxt.__author__,
    author_email=pxt.__email__,
    long_description=pxt.__description__,
    description=pxt.__description__,
    classifiers=classifiers,
    install_requires=requirements,
    dependency_links=['https://github.com/havocesp/cctf.git'],
)
