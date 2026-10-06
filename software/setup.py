"""Tag the ctypes native library wheel for its target operating system."""
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel

class NativeLibraryWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform = super().get_tag()
        return 'py3', 'none', platform

setup(cmdclass={'bdist_wheel': NativeLibraryWheel})
