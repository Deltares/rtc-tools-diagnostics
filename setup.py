"""Toolbox for diagnostics for RTC-Tools

This toolbox includes several utilities to analyze the results of an RTC-Tools optimization run.
"""
from setuptools import setup, find_packages
import versioneer

DOCLINES = __doc__.split("\n")


setup(
    name='rtc-tools-diagnostics',
    version=versioneer.get_version(),
    maintainer='Deltares',
    author='Deltares',
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    platforms=['Windows', 'Linux', 'Mac OS-X', 'Unix'],
    install_requires=["rtc-tools >= 2.5.0",
                      "tabulate", "casadi", "numpy", "pandas"],
    tests_require=['pytest', 'pytest-runner'],
    python_requires='>=3.5',
    cmdclass=versioneer.get_cmdclass(),
)
