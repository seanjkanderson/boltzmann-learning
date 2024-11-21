from setuptools import setup, find_packages

setup(
    name="LearningGames",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy>=1.26.4',
        'matplotlib>=3.9.2',
        'scipy>=1.13.1',
        'pandas>=2.2.2',
        'scikit-learn>=1.5.2',
        'lightgbm>=4.5.0'
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
        ],
    },
    include_package_data=True,
    python_requires=">=3.9",
)