from setuptools import find_packages, setup

setup(
    name="cooperative-planner",
    version="0.0",
    description="Sampling-based motion planner for human-robot cooperative table-carrying",
    url="https://github.com/eleyng/cooperative-planner",
    author="eleyng",
    author_email="eleyng@stanford.edu",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "scipy",
        "pytorch-lightning",
        "numpy",
        "matplotlib",
        "tqdm",
        "wandb",
    ],
)
