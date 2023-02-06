from pathlib import Path
from setuptools import setup, Extension, find_packages
from distutils.command.build_ext import build_ext


root = Path(__file__).parent

with (root / "curated_transformers" / "about.py").open("r") as f:
    about = {}
    exec(f.read(), about)


def setup_package():
    setup(
        name="curated-transformers",
        version=about["__version__"],
        packages=find_packages(),
    )


if __name__ == "__main__":
    setup_package()
