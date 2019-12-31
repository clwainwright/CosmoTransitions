from setuptools import setup

setup(
    name="cosmoTransitions",
    version="2.0.5",
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
    license="MIT",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],
)
