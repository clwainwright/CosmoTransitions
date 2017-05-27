from setuptools import setup

setup(
    name="cosmoTransitions",
    version="2.0.1",
    packages=['cosmoTransitions', 'cosmoTransitions.examples'],
    package_dir={'cosmoTransitions.examples': 'examples'},
    description=(
        "A package for analyzing finite or zero-temperature cosmological "
        "phase transitions driven by single or multiple scalar fields."
    ),
    author="Carroll L. Wainwright",
    author_email="clwainwri@gmail.com",
    url="https://github.com/clwainwright/CosmoTransitions",
    install_requires=['numpy>=1.8', 'scipy>=0.11'],
)
