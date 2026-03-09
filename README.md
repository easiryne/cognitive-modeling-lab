# Undertanding Human Behavior with Cognitive Models - Travaux Dirigés (TD)
## Table of Contents
### [TD 1 - The Bandit task](TD_1)
   Get familiar with the **two-armed Bandit task**. Simulate behavior with a **computational model** and understand the role of **parameters**. Do the task and fit the model to your own behavior. 

   Download: &nbsp; [Exercise files 📦](TD_1/TD_1.zip) &nbsp;&nbsp;|&nbsp;&nbsp; [Solution 📝](TD_1/td1_solution.ipynb)

### [TD 2 - Drift-Diffusion Model](TD_2)
   Understand how to use the **Drift-diffusion model** to analyze behavioral data from the **Random-dot motion task**. Fit descriptive equations for reaction time and accuracy as a function of motion coherence. For this, learn how to define a **cost function**.

   Download: &nbsp; [Exercise files 📦](TD_2/TD_2.zip) &nbsp;&nbsp;|&nbsp;&nbsp; [Solution 📝](TD_2/td2_solution.ipynb)

### [TD 3 - Signal Detection Theory](TD_3)
   Get familiar with model comparison and falsification in the **Stimulus Detection Task**. Implement the **High-Threshold Theory (HTT)** and **Signal Detection Theory (SDT)** models, and use **ROC curves** to compare them.

   Download: &nbsp; [Exercise files 📦](TD_3/TD_3.zip) &nbsp;&nbsp;|&nbsp;&nbsp; [Solution 📝](TD_3/td3_solution.ipynb)

### [TD 4 - Fifty Shades of Bandit](TD_4)
   Explore variations of the **Bandit task**: stationary, reversal, and restless. Implement a **Reinforcement Learning** model and optimize its parameters for different task variants.

   Download: &nbsp; [Exercise files 📦](TD_4/TD_4.zip) &nbsp;&nbsp;|&nbsp;&nbsp; [Solution 📝](TD_4/td4_solution.ipynb)

### [TD 5 - Likelihood and Parameter Estimation](TD_5)
   Understand how to compute the **likelihood** of a behavioral trajectory given a computational model. Explore how model parameters influence the likelihood and implement **Maximum Likelihood Estimation (MLE)**.

   Download: &nbsp; [Exercise files 📦](TD_5/TD_5.zip) &nbsp;&nbsp;|&nbsp;&nbsp; [Solution 📝](TD_5/td5_solution.ipynb)

### [TD 6 - Parameter Recovery](TD_6)
   Learn how to recover the parameters of a cognitive model from behavioral data. Implement **parameter recovery** analysis, visualize the results, and produce a **confusion matrix**.

   Download: &nbsp; [Exercise files 📦](TD_6/TD_6.zip) &nbsp;&nbsp;|&nbsp;&nbsp; [Solution 📝](TD_6/td6_solution.ipynb)

### [TD 7 - Model Recovery](TD_7)
   Implement **model recovery** on several computational models for the Bandit task. Implement the complete model recovery analysis and report the results in an **inversion matrix**.

   Download: &nbsp; [Exercise files 📦](TD_7/TD_7.zip) &nbsp;&nbsp;|&nbsp;&nbsp; [Solution 📝](TD_7/td7_solution.ipynb)

### [TD 8 - Model Selection and Cross-Validation](TD_8)
   Learn about **model selection** on experimental data from the reversal Bandit task. Fit multiple RL models to participant data, compare them using **information criteria**, and apply **cross-validation** techniques.

   Download: &nbsp; [Exercise files 📦](TD_8/TD_8.zip) &nbsp;&nbsp;|&nbsp;&nbsp; [Solution 📝](TD_8/td8_solution.ipynb)

### [TD 9 - Bayesian Inference](TD_9)
   Understand the basic principles of **Bayesian inference**: **prior**, **likelihood**, and **posterior**. Apply **Bayes' rule** to discrete and continuous distributions. Implement a **Bayesian RL** model and apply it to the bandit task.

   Download: &nbsp; [Exercise files 📦](TD_9/TD_9.zip) &nbsp;&nbsp;|&nbsp;&nbsp; [Solution 📝](TD_9/td9_solution.ipynb)

### [TD 10 - Sequential Perception and Review](TD_10)
   Work on a **sequential perception task**. Implement two models, simulate behavior, and run the full pipeline: parameter recovery, model recovery, model fitting, and model selection (fixed and random effects).

   Download: &nbsp; [Exercise files 📦](TD_10/TD_10.zip) &nbsp;&nbsp;|&nbsp;&nbsp; [Solution 📝](TD_10/td10_solution.ipynb)

<hr style="height:1px; border:none; background:#333;" />

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