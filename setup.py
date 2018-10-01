# -*- coding:utf-8 -*-
from pathlib import Path

from setuptools import setup, find_packages

from pandaxt import (__version__, __author__, __description__, __site__, __email__, __license__, __keywords__,
                     __package__)

requirements = Path.cwd() / 'requirements.txt'
requirements = requirements.read_text(encoding='utf8')
requirements = requirements.split('\n')

exclude = ['.idea*', 'build*', '{}.egg-info*'.format(__package__), 'dist*', 'venv*', 'doc*', 'lab*']

classifiers = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]

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
    classifiers=classifiers,
    install_requires=requirements
)
