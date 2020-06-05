from setuptools import setup, find_packages


VERSION = '0.0.2'

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')


setup(
    name='eisen-deploy',
    version=VERSION,
    description='Eisen deploy provides model deployment and serving functionality',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [],
    },
)
