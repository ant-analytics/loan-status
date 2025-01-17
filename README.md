To install the required packages, use the `requirements.txt` file:
```bash
pip install -r requirement_pip.txt
```

Alternatively, you can use Conda to install the packages:

```bash
conda install --file requirement_codna.txt
```

You can find the full report in the following Jupyter Notebook: [notebooks/report.ipynb](notebooks/report.ipynb)

For TensorBoard visualization, run the following command:

```bash
tensorboard --logdir=logs
```
## Introduction

This project aims to analyze and predict loan status and credit score of potential clients using a neural network with two outputs. The goal is to identify patterns and factors that influence loan approval and credit scores, and to build a predictive model that can accurately classify the loan status and predict the credit score of new applicants.
