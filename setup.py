from setuptools import setup
import versioneer


setup(
    name='rtc-tools-diagnostics',
    version=versioneer.get_version(),
    maintainer='Deltares',
    author='Deltares',
    description="Toolbox for diagnostics for RTC-Tools",
    install_requires=["rtc-tools >= 2.5.0",
                      "tabulate", "casadi", "numpy", "pandas"],
    tests_require=['pytest', 'pytest-runner'],
    python_requires='>=3.5',
    cmdclass=versioneer.get_cmdclass(),
)
