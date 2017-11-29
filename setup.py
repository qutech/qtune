import subprocess
import setuptools
import os.path
import re

from pip.req import parse_requirements


def get_file(*args):
    return os.path.join(os.path.dirname(__file__), *args)


def read(*args):
    return open(get_file(*args)).read()


def get_version():
    version_match = re.search(r"^__version__\s=\s['\"]([^'\"]*)['\"]", read('qtune', '__init__.py'), re.M)

    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError('Could not extract version from {}'.format(get_file('qtune', '__init__.py')))


def is_matlab_engine_installed():
    try:
        import matlab.engine
        return True
    except ImportError:
        return False


def find_matlab_engine_installer_dir():
    p = subprocess.Popen(['where', 'matlab'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate(timeout=1)

    if p.returncode:
        raise RuntimeError('Could not locate matlab', str(err, 'utf-8'))
    else:
        matlab_root = os.path.dirname(str(out, 'utf-8').splitlines()[0])

        matlab_engine_dir = os.path.join(matlab_root, 'extern', 'engines', 'python')

        matlab_engine_installer = os.path.join(matlab_engine_dir, 'setup.py')
        if not os.path.exists(matlab_engine_installer):
            raise RuntimeError('Could not find matlab engine installer')

        return matlab_engine_dir


def install_matlab():
    if not is_matlab_engine_installed():
        matlab_installer_dir = find_matlab_engine_installer_dir()

        matlab_engine_installer = os.path.join(matlab_installer_dir, 'setup.py')

        install_process = subprocess.Popen(['python', matlab_engine_installer, 'install'], cwd=matlab_installer_dir)
        install_process.communicate()

        if install_process.returncode:
            raise RuntimeError('Error while installing matlab engine')


def get_requirements():
    return [str(ir.req) for ir in parse_requirements(get_file('requirements.txt'), session=True)]


setuptools.setup(
    name="qtune",
    version=get_version(),
    author="RWTH Aachen Quantum Technology Group",
    author_email="simon.humpohl@rwth-aachen.de",
    keywords="autotune quantum",
    url="https://git.rwth-aachen.de/qutech/python-atune",
    packages=['qtune'],
    package_data={'qtune': ['qtune/MATLAB/*/*.m']},
    long_description=read('README'),
    install_requires=get_requirements()
)
