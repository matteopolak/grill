from setuptools import setup, find_packages

def get_requirements(path: str):
    return [l.strip() for l in open(path)]

setup(
    name='grill',
    version='0.0.1',
    packages=find_packages(),
    requires=get_requirements('requirements.txt'),
)

