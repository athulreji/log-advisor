from setuptools import setup
setup(
    name='log-advisor',
    version='0.0.1',
    entry_points={
        'console_scripts': [
            'log-advisor=main:run'
        ]
    }
)