import os

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            return line.split("'")[1]
    raise RuntimeError('Unable to find version string.')

with open('requirements.txt', 'r') as requirements:
    setup(name='rl_jailbreak',
        version=get_version('rl_jailbreak/__init__.py'),
        install_requires=list(requirements.read().splitlines()),
        packages=find_packages(),
        description='Library for training LLMs with RL to generate universal LLM jailbreaks',
        python_requires='>=3.8',
        author='Xavier Yin and Charlie Ji',
        author_email='nzxyin@berkeley.edu',
        # TODO add classifiers and license stuff
        # classifiers=[
        #     'Programming Language :: Python :: 3',
        #     'License :: OSI Approved :: MIT License',
        #     'Operating System :: OS Independent'
        # ],
        long_description=long_description,
        long_description_content_type='text/markdown')