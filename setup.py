from setuptools import find_packages, setup

setup(
    name='pyCascade',
    packages=find_packages(include=['pyCascade']),
    version='0.1.0',
    description='python library to complement Cascade simulation tools',
    author='Nicholas Bachand',
    license='MIT',
    install_requires=['glob2', 'numpy', 'pandas', 'matplotlib', 'pandarallel', 'opencv-python'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)