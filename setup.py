import subprocess
import setuptools
import os.path
import re
import sys


REQUIRED_PACKAGES = [
    'pandas',
    'numpy',
    'filterpy',
    'scipy',
    'h5py',
    'sympy'
]


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
        try:
            encoding = re.findall(rb'(\d+)', subprocess.check_output('chcp.com'))[0].decode()
        except:
            encoding = 'utf-8'

        try:
            path = subprocess.check_output(['where', 'matlab']).decode(encoding).splitlines()[0]
        except subprocess.CalledProcessError:
            raise RuntimeError('Could not locate matlab')

        return os.path.dirname(os.path.dirname(path))

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


setuptools.setup(
    name="qtune",
    version=get_version(),
    
    author="RWTH Aachen Quantum Technology Group",
    author_email="julian.teske@rwth-aachen.de",
    
    keywords="autotune quantum",
    url="https://github.com/qutech/qtune",
    
    packages=['qtune'],
    package_data={'qtune': ['qtune/MATLAB/*/*.m']},
    
    license="GNU GPLv3+",

    description="Quantum dot auto tune framework",
    
    long_description=read('README.md'),
    long_description_content_type="text/markdown",

    install_requires=REQUIRED_PACKAGES,
    setup_requires=['pytest-runner'] + REQUIRED_PACKAGES,

    test_suite="tests",
    tests_require=['pytest'] + REQUIRED_PACKAGES,

    cmdclass={'install_matlab': MatlabInstall},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)"    
    ]
)
