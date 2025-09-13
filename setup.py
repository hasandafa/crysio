from setuptools import setup, find_packages

setup(
    name="crysio",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    package_data={
        'crysio': ['data/sample_structures/*'],
    },
)