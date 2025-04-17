from setuptools import setup, find_packages

setup(
    name="seismic-wave-visualizer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'PyQt5',
        'obspy',
        'pandas',
        'pyqtgraph'
    ],
    entry_points={
        'console_scripts': [
            'seismic-visualizer=src.main:main',
        ],
    },
)