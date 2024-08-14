import os
import re
import subprocess as sp
from pathlib import Path

from dotenv import load_dotenv
from setuptools import find_packages, setup
from setuptools.command.develop import develop as _develop

load_dotenv()
NEXUS_PYPI_URL = os.getenv("NEXUS_WEIGHTS_URL", default="aivdl.nexus.aiv.ai/repository/aiv-pypi/simple")
NEXUS_USER = os.getenv("NEXUS_USER", default="download")
NEXUS_PASSWORD = os.getenv("NEXUS_PASSWORD", default="1234")

FILE = Path(__file__).resolve()
PARENT = FILE.parent
README = (PARENT / 'README.md').read_text(encoding='utf-8')

REQ_TO_IGNORE = ['tensorboard', 'tensorflow', 'torch']
REQ_TO_INCLUDE = ['aivcommon', 'aivocr']


def check_req_to_ignore(line, ignore_base=True):

    for req in REQ_TO_IGNORE + parse_base_requirements():
        if req in line:
            return False
    return True


def get_version():
    file = PARENT / 'openvisionsuite/__init__.py'
    version = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]

    try:
        version = sp.check_output(["git", "describe", "--tags", "--abbrev=0"]).decode().strip()[1:]
        return os.environ.get('PACKAGE_VERSION', version)  # Default version if git tag retrieval fails
    except Exception as err:
        print("Error:", err)
        return os.environ.get('PACKAGE_VERSION', version)  # Default version if git tag retrieval fails


def parse_base_requirements(file_path=PARENT / 'requirements/base_requirements.txt'):

    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()

        if line and not line.startswith('#'):
            try:
                _line = line.split("#")[0].strip().split("==")[0]
            except Exception as err:
                print(err)
                _line = line.split("#")[0].strip()

            requirements.append(_line)

    return requirements


def parse_requirements(file_path: Path):

    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()

        if line and not line.startswith('#') and not line.startswith('-e') and not line.startswith(
                '--') and check_req_to_ignore(line):
            requirements.append(line.split("#")[0].strip())

    print("====================== requirements ========================")
    print(requirements)
    print("============================================================")

    return requirements


class CustomDevelopCommand(_develop):

    def run(self):
        # Install the package from Nexus
        for lib in REQ_TO_INCLUDE:
            sp.check_call([
                "pip", "install", lib, f"--extra-index-url=http://{NEXUS_USER}:{NEXUS_PASSWORD}@{NEXUS_PYPI_URL}",
                "--trusted-host=aivdl.nexus.aiv.ai", "--force-reinstall"])
        _develop.run(self)


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

