from setuptools import setup, find_packages

setup(
    name="LearningGames",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy', 'matplotlib', 'scipy'
    ],
    extras_require={
        "dev": [
            "pytest",
            # Add other development dependencies here
        ],
    },
    entry_points={
        "console_scripts": [
            # Define any command-line scripts here
            # e.g., 'command_name=module:function'
        ],
    },
    include_package_data=True,
    python_requires=">=3.6",
)