from sphinx.setup_command import BuildDoc
from setuptools import setup

cmdclass = {'build_sphinx': BuildDoc}

version = '0.1'
setup(
    cmdclass=cmdclass,
    platforms=["Linux"],
    python_requires=">=3.7",
    # these override conf.py settings
    command_options={
        'build_sphinx': {
            'version': ('setup.py', version),
            'source_dir': ('setup.py', 'docs')}},
    extras_require={
        "build_docs": ["sphinx>=3", "sphinx-rtd-theme"]},
)

