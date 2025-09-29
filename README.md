# Undertanding Human Behavior with Cognitive Models - Travaux Dirigés (TD)

## Installation
⚠️ Try to you complete the installation before the first TD session to save time!  

### 1-Download the files ⬇️
- Create a folder on your computer where you will store all TD files.
- Download the attached files `environment.yml` and `td0_test_installation.ipynb` and place them in the folder you just created.

---

### 2- Install Python and libraries 🐍
#### A) **ADVANCED**: If you already have Python installed (and you are familiar with the command-line terminal)   

You will need a virtual environment with Python 3.11 and the dependencies listed `environment.yml`. You can create the new environment named `tdenv` with this command:

```shell
conda env create -f environment.yml
```

---

#### B) **BEGINNER**: If you need to install Python on your computer  

I advise installing it with the [Anaconda distribution.](https://www.anaconda.com/download/success)  

Anaconda takes up about 4 GB on your hard drive. If you are concerned about storage on your computer, 
you can also download [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/), which is a smaller Python distribution that contains very few pre-installed packages, 
and doesn't come with Anaconda's user interface (**Anaconda Navigator**). 
If you prefer to avoid using command-line terminal, and especially if your computer runs on Windows, then Anaconda is a better choice for you.

Once you have installed Anaconda:
1. Open the **Anaconda Navigator** application
2. In the `Environments` 📦 tab (left side bar), create a new **conda environment** for the TDs:
   - Click on the `Import` button at the bottom of the Environment list
   - Select the `environment.yml` file that you downloaded previously
   - Name the new environment `tdenv`
   - Wait for the environment to be created. All the necessary libraries will be automatically installed.

---

### 3- Testing your installation 🎯
All our TDs will be done on Jupyter Notebooks. The `JupyterLab` app is appropriate for the work we will do. To check that you are ready for the first TD, run the `td0_test_installation.ipynb` notebook in JupyterLab:

1. In the Anaconda Navigator, go back to the `Home`🏠 tab, install and launch the `JupyterLab` app. It will open in your browser.
2. In JupyterLab, find the `td0_test_installation.ipynb` file where you placed it and double-click on it.
3. The **TD0 - Setup** notebook will open. Just run the first code cell (with the imports) to check if the installed packages are working.

**ADVANCED**: To launch JupyterLab with the terminal, go back to the directory you created, activate the conda environment and launch JupyterLab:
```shell
conda activate tdenv
jupyter lab
```