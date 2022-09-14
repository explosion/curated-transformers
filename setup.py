from setuptools import setup, Extension, find_packages
from distutils.command.build_ext import build_ext


def setup_package():
    setup(
        name="curated-transformers",
        packages=find_packages(),
    )


if __name__ == "__main__":
    setup_package()
