# -*- coding:utf-8 -*-
"""Setup script module"""
from pathlib import Path

from setuptools import find_packages, setup

import pandaxt as pxt

requirements = Path('requirements.txt').read_text(
    encoding='utf-8').splitlines()

exclude = ['.idea*', 'build*',
           f'{pxt.__package__}.egg-info*', 'dist*', 'venv*', 'doc*', 'lab*']

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11'
]

setup(
    name=pxt.__package__,
    version=pxt.__version__,
    packages=['pandaxt'] + find_packages(exclude=exclude),
    url=pxt.__site__,
    license=pxt.__license__,
    keywords=pxt.__keywords__,
    author=pxt.__author__,
    long_description=Path('README.md').read_text(),
    author_email=pxt.__email__,
    description=pxt.__description__,
    long_description_content_type='text/markdown',
    classifiers=classifiers,
    install_requires=requirements
)
