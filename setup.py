import os
import re
import subprocess as sp
from pathlib import Path

from dotenv import load_dotenv
from setuptools import find_packages, setup
from setuptools.command.develop import develop as _develop

FILE = Path(__file__).resolve()
PARENT = FILE.parent
README = (PARENT / 'README.md').read_text(encoding='utf-8')


def get_version():
    file = PARENT / 'openvisionsuite/__init__.py'
    version = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]

    try:
        version = sp.check_output(["git", "describe", "--tags", "--abbrev=0"]).decode().strip()[1:]
        return os.environ.get('PACKAGE_VERSION', version)  # Default version if git tag retrieval fails
    except Exception as err:
        print("Error:", err)
        return os.environ.get('PACKAGE_VERSION', version)  # Default version if git tag retrieval fails



setup(
    name='OpenVisionSuite',
    version='{{VERSION_PLACEHOLDER}}',
    python_requires='>=3.9',
    description=('Template for Python openvisionsuite'),
    long_description=README,
    long_description_content_type='text/markdown',
    # packages=['openvisionsuite'] + [str(x) for x in Path('openvisionsuite').rglob('*/') if x.is_dir() and '__' not in str(x)],
    packages=find_packages(exclude=[]),
    package_data={
        '': ['*.yaml', '*.json'], },
    include_package_data=True,
    # install_requires=parse_requirements(PARENT / 'requirements/requirements.txt'),
    # cmdclass={
    #     'develop': CustomDevelopCommand, }
)

