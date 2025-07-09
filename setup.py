from setuptools import setup

setup(
    name="microdf-python",
    version="0.4.4",
    description="Survey microdata as DataFrames.",
    url="http://github.com/PSLmodels/microdf",
    author="Max Ghenis",
    author_email="max@ubicenter.org",
    license="MIT",
    packages=["microdf"],
    install_requires=[
        "numpy",
        "pandas",
    ],
    extras_require={
      "taxcalc": ["taxcalc"],
      "charts": [
        "seaborn",
        "matplotlib",
        "matplotlib-label-lines"
      ]
    },
    zip_safe=False,
)
