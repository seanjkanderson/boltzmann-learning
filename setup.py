from setuptools import setup, find_packages

setup(
    name="LearningGames",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'pandas',
        'scikit-learn',
        'lightgbm'
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
    data_files=[('/data/', []), ('/ember_data/', [])],
    include_package_data=True,
    python_requires=">=3.9",
)