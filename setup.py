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
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'pandas==1.2.*',
        'scipy',
        'plotly',
        'pyedflib',
        'scikit-image',
        'pywavelets',
        'networkx',
        'dateparser',
        'unidecode',
        'mne==0.22.*',
        'h5py==3.2.*',
    ]
)
