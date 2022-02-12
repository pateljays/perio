Deployment of PD prediction models

****** Requirement

- Python 3

Python packages:
- pandas
- numpy
- matplotlib
- shap   https://github.com/slundberg/shap
  install shpa using:
  pip install shap
  or
  conda install -c conda-forge shap



****** Input

 Please format your input following _target_example.csv, missing value should be blank

 More samples can be found in _test_random_42.csv file.


****** Run

 Run prediction_tool.py file


****** Parameters
- selected_patients: a list of IDs of selected patients.



****** Output

 Output files can be found in prediction_res folder.

 You will find an Instruction.doc file, which provides guidance on how to read the results. 

 - predicted_risk.csv: predicted risk scores of selected patients
 - evidence_[ID].pdf: illustration of contributions of predictors
 - evidence_[ID].csv: detailed SHAP values, i.e., contributions of predictors



