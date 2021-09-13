from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as fh:
    license = fh.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='Correlation Analysis in High Dimensional Data',
    version='0.1.0',
    description='Tools to perform correlation analysis',
    long_description=readme,
    author='Praneeth Balakrishna',
    #url='https://github.com/',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=requirements,
    python_requires='>=3.7'
)


