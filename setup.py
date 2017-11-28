import subprocess
import setuptools
import os.path
import re
import sys

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


class MatlabInstall(setuptools.Command):
    description = "Install MATLAB engine"
    user_options = [('matlabroot=', None, 'path to MATLAB version to install')]

    @staticmethod
    def is_engine_installed():
        try:
            import matlab.engine
            return True
        except ImportError:
            return False

    @staticmethod
    def find_matlabroot():
        p = subprocess.Popen(['where', 'matlab'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(timeout=1)

        if p.returncode:
            raise RuntimeError('Could not locate matlab', str(err, 'utf-8'))
        else:
            return os.path.dirname(str(out, 'utf-8').splitlines()[0])

    def get_installer_dir(self):
        return os.path.join(self.matlabroot, 'extern', 'engines', 'python')

    def get_installer_script(self):
        return os.path.join(self.get_installer_dir(), 'setup.py')

    def initialize_options(self):
        self.matlabroot = None

    def finalize_options(self):
        if self.matlabroot:
            assert os.path.exists(self.matlabroot), 'Provided MATLAB root path does not exist'
        else:
            self.matlabroot = self.find_matlabroot()

        assert os.path.isfile(self.get_installer_script()), '{} does not exist'.format(self.get_installer_script())

    def run(self):
        if not self.is_engine_installed():
            install_process = subprocess.Popen([sys.executable, self.get_installer_script(), 'install'],
                                               cwd=self.get_installer_dir())
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
    long_description=read('README'),
    install_requires=get_requirements(),

    cmdclass={'install_matlab': MatlabInstall}
)
