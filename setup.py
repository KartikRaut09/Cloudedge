from setuptools import setup, find_packages

setup(
    name="cloudedge",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "openenv-core",
        "fastapi",
        "uvicorn",
        "pydantic>=2.0",
        "matplotlib",
        "numpy",
    ],
)
