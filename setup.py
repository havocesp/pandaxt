# -*- coding:utf-8 -*-
from setuptools import setup, find_packages

from pandaxt import (__version__, __author__, __description__, __site__, __email__, __license__, __keywords__,
                     __package__)

exclude = ['.idea*', 'build*', '{}.egg-info*'.format(__package__), 'dist*', 'venv*', 'doc*', 'lab*']

setup(
    name=__package__,
    version=__version__,
    packages=find_packages(exclude=exclude),
    url=__site__,
    license=__license__,
    packages_dir={'': __package__},
    keywords=__keywords__,
    author=__author__,
    author_email=__email__,
    long_description=__description__,
    description=__description__,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        'pandas',
        'tulipy',
        'ccxt'
    ]
)
