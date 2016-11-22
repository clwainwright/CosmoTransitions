from setuptools import setup


from cosmoTransitions import __version__

setup(
    name='cosmoTransitions',
    version=__version__,
    url='https://github.com/clwainwright/CosmoTransitions',
    description='A package for analyzing finite or zero-temperature cosmological phase transitions driven by single or multiple scalar fields.',
    packages=['cosmoTransitions'],
    install_requires=[
        "numpy",
        "scipy",
    ],
)