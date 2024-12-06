# Computational Cognitive Modeling - Travaux Dirigés

## Installation
⚠️ Make sure you complete this part before the first TD session!  

### Installing Python
You need to install Python on your machine. I advise installing it with the [Anaconda distribution.](https://www.anaconda.com/download/success)  

Anaconda takes up about 4 GB on your hard drive. If you are concerned about storage on your computer, you can also download [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/), which is a smaller distribution of Python that contains very few pre-installed packages, and doesn't come with Anaconda's user interface. If you don't like using the command-line terminal, then Anaconda is a better choice for you.

The Python version shouldn't matter too much for the exercises, but you might as well get the latest version.

### Installing the required packages
Make sure the packages listed in `requirement.txt` are installed in your working Python environment. You can install them all with the following command:
```shell
conda install --yes --file requirements.txt
```