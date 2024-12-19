# Beta
Machine learning projects
house_price_prediction/
│
├── data/              # Raw and cleaned datasets
│   ├── raw_data.csv
│   └── cleaned_data.csv
│
├── notebooks/         # Jupyter Notebooks for exploration and model building
│   ├── 1_EDA.ipynb
│   ├── 2_Modeling.ipynb
│   └── 3_Evaluation.ipynb
│
├── reports/           # Reports and presentations
│   └── final_report.pdf
│
├── src/               # Python scripts for modular code
│   ├── data_cleaning.py
│   ├── model_training.py
│   └── model_evaluation.py
|   |__ trained_model.pkl # save model after training
│
├── README.md          # Project description
├── requirements.txt   # Libraries needed
└── main.py            # Main script to execute the project

3. Data Usage
The dataset can be used for multiple purposes:

Exploratory Data Analysis (EDA): Analyze key features, distribution patterns, and relationships to understand credit risk factors.
Classification: Build predictive models to classify the loan_status variable (approved/not approved) for potential applicants.
Regression: Develop regression models to predict the credit_score variable based on individual and loan-related attributes.
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data?select=loan_data.csv
