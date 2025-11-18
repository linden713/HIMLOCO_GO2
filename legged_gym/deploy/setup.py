from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent

setup(
    name="legged-gym-deploy",
    version="0.1.0",
    description="Deployment utilities for running legged gym policies on Unitree hardware.",
    long_description=(ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(include=["deploy_real", "deploy_real.*"]),
    package_data={"deploy_real": ["configs/*.yaml"]},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "onnxruntime",
        "pyyaml",
        "unitree-sdk2py",
    ],
)
