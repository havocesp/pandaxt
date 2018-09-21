from setuptools import setup
from pandaxt import __version__, __author__, __description__, __site__, __license__, __appname__,

setup(
    name=__appname__,
    version=__version__,
    packages=[''],
    package_dir={'': __package__},
    url=__site__,
    license=__license__,
    author=__author__,
    author_email='',
    description=__description__
)
