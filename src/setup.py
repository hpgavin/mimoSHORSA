
# file setup.py  ------------------------------

from setuptools import setup, find_packages

setup(
    name='mimoSHORSA',
    version='1.0',
    packages=find_packages(),
    install_requires=[])


## in console -----------------------------------

# sudo apt install pipx
# cd mimoSHORSA/src
# python3 setup.py sdist bdist_wheel
# pipx install -e .

