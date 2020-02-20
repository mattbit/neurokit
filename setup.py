import setuptools

setuptools.setup(
    name='Neurokit',
    version='0.0.1-alpha',
    author='Matteo Dora',
    author_email='matteo.dora@ens.psl.eu',
    description='Neurophysiological timeseries toolkit',
    url='https://github.com/mattbit/neurokit',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'fastparquet',
        'plotly',
        'pyedflib',
        'skimage',
        'pywavelets',
        'networkx',
        'dateparser',
        'chardet',
        'unidecode'
    ]
)
