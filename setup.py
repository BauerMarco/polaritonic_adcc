from sphinx.setup_command import BuildDoc
from setuptools import setup

cmdclass = {'build_sphinx': BuildDoc}

setup(
    cmdclass=cmdclass,
    platforms=["Linux"],
    install_requires=[
        "adcc >= 0.15.17",
    ],
    python_requires=">=3.11",
    # these override conf.py settings
    # command_options={
    #    'build_sphinx': {
    #        'version': ('setup.py', version),
    #        'source_dir': ('setup.py', 'docs')}},
)

