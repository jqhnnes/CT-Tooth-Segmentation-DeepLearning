"""Setup script for 3D Tooth Segmentation package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tooth-segmentation-3d",
    version="1.0.0",
    author="Johannes",
    description="3D Tooth Segmentation using Deep Learning for µCT Scans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jqhnnes/Segmentierung-von-CT-Aufnahmen-extrahierter-Z-hne-mittels-Deep-Learning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tooth-train=training.train:main",
            "tooth-evaluate=evaluation.evaluate:main",
        ],
    },
)
