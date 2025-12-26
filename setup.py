"""Setup script for automated-time-series-forecasting."""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='automated-time-series-forecasting',
    version='1.0.0',
    author='Tharun Ponnam',
    author_email='tharunponnam007@gmail.com',
    description='Automated time series forecasting with ARIMA, Exponential Smoothing, and LSTM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tharun-ship-it/automated-time-series-forecasting',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'statsmodels>=0.13.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'pyyaml>=6.0',
    ],
    extras_require={
        'deep_learning': ['tensorflow>=2.8.0'],
        'dev': ['pytest', 'black', 'flake8'],
        'all': ['tensorflow>=2.8.0', 'plotly>=5.0.0', 'seaborn>=0.11.0'],
    },
    entry_points={
        'console_scripts': [
            'ts-forecast=src.pipeline.forecaster:main',
        ],
    },
)
