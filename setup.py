from setuptools import setup

setup(
    name="cosmoTransitions",
    version="2.0.6",
    packages=['cosmoTransitions', 'cosmoTransitions.examples'],
    package_dir={'cosmoTransitions.examples': 'examples'},
    description=(
        "A package for analyzing finite or zero-temperature cosmological "
        "phase transitions driven by single or multiple scalar fields."
    ),
    author="Carroll L. Wainwright",
    author_email="clwainwri@gmail.com",
    url="https://github.com/clwainwright/CosmoTransitions",
    python_requires='>3.7',
    install_requires=['numpy>=1.16.5', 'scipy>=1.6'],
    license="MIT",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
