#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:21:17 2022

@author: changsu
"""


import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import pickle
from io import BytesIO
import base64

def predict_tool(file_dir, p_info):
    '''
    file_dir:   dir of files used in following process
    p_info:     pd dataframe of patient info
    
    '''
    
    CAT_FEATURES = [
        'Race_white', 'Race_black', 'Race_asian', 'Race_native', 'Race_multiracial',
        'Sex_female', 'Language_ENG', 'Insurance_Cash', 'Insurance_Private', 'Insurance_Medicaid', 'Insurance_Ryan_White',
        # 'AgeBaseline',
        'Med1', 'Med2', 'Med3', 'Med4', 'Med5', 'Med6', 'Med7', 'Med8', 'Med9', 'Med10', 'Med11', 'Med12',
        'CRA', 'Radiographs', 'Tobacco', 'Alcohol', 'RecDrugs', 'Preg999t', 'ASA',
        # 'DMFT', 'DMFS', 'Teeth', 'PlaqueIdx', 'BOP',
        'BoneLoss', 'BL_URQ', 'BL_LRQ', 'BL_ULQ', 'BL_LLQ', 'BL_VERT',
        # 'BL_MaxPct',
        'Calc_URQ', 'Calc_LRQ', 'Calc_ULQ', 'Calc_LLQ',
        'RFG1', 'RFG2', 'RFG3', 'RFG4', 'RFG5', 'RFG6', 'RFG7', 'RFG8',
        'RFL1', 'RFL2', 'RFL3', 'RFL4', 'RFL5', 'RFL6', 'RFL7', 'RFL8', 'RFL9', 'RFL10',
        'DentalHealth', 'PainLevel', 'Chewing', 'Speaking', 'SelfImage',
        'AnxietyYest', 'AnxietyToday', 'ClenchGrind', 'GumTrouble', 'BrushFreq', 'FlossFreq',
        'Denture', 'SRPSxPerio', 'OralSx', 'Trauma', 'OralCancer', 'TMJ', 'Ortho',
        # 'PerioDx', 'PerioDx_new', 'StudyId'
    ]

    CONT_FEATURES = [
        'AgeBaseline',
        'DMFT', 'DMFS', 'Teeth', 'PlaqueIdx', 'BOP',
        'BL_MaxPct',
    ]

    #outcome_col = 'PerioDx_new'
    '''
    selected_patients = [9461] # IDs of selected patients

    # load data to predict
    target_data = pd.read_csv('_target_example.csv', header=0)
    target_data = target_data[target_data['StudyId'].isin(selected_patients)]
    '''
    target_data=p_info 
    
    # load imputer
    with open(file_dir+'/'+'_cont_imputer.pkl' , 'rb') as f:
        imp_median = pickle.load(f)
    
    with open(file_dir+'/'+'_cat_imputer.pkl' , 'rb') as f:
        imp_freq = pickle.load(f)
  
    # Imputation
    data_cont_imputed = imp_median.transform(target_data[CONT_FEATURES])
    data_cat_imputed = imp_freq.transform(target_data[CAT_FEATURES])
    data_imputed = pd.DataFrame(data=np.concatenate((data_cat_imputed, data_cont_imputed, target_data[['StudyId']].values), axis=1),
                                columns=CAT_FEATURES + CONT_FEATURES + ['StudyId'])

    #print('Imputation finished...')


    # load model
 
    with open(file_dir+'/'+'_model_pkl.pkl' , 'rb') as f:
        model = pickle.load(f)

    model_severe_PD = model.estimators_[2] # separating Severe PD vs others

    X_data = data_imputed[CAT_FEATURES + CONT_FEATURES].values
    

    y_pred = model_severe_PD.predict_proba(X_data)[:, 1]

    y_pred_norm = 100 * (np.clip(y_pred, 0, 0.7) - 0) / (0.7 - 0)

    y_pred_df = pd.DataFrame(columns=['StudyId', 'Severe_PD_risk'])
    y_pred_df['StudyId'] = target_data['StudyId'].values
    #y_pred_df.insert(1, "Severe_PD_risk", y_pred_norm, True)
    y_pred_df['Severe_PD_risk'] = y_pred_norm

    # individual prediction interpretation
    save_dir = 'prediction_res'
    features = CAT_FEATURES + CONT_FEATURES
    for idx in range(len(target_data)):
        p_id = target_data.loc[idx, 'StudyId']
        #print(idx, p_id)
        
        shap.initjs()
        explainer = shap.TreeExplainer(
                                       model = model_severe_PD,
        #                               data = shap.sample(train_imputed[features], 100), 
                                       model_output='raw', #'probability'
                                       )
    
        shap_values_val = explainer.shap_values(data_imputed[features])
        plt.switch_backend('Agg')
        plt.figure(figsize=(20, 4))
        shap.force_plot(
                #explainer.expected_value, 
                0,
                shap_values_val[idx, :], 
                data_imputed[features].iloc[idx, :],
                matplotlib=True,
                show=False
                )
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        shap_plot = base64.b64encode(image_png)
        shap_plot = shap_plot.decode('utf-8')
        plt.close()
        #plt.savefig(save_dir + '/evidence_%s.pdf' % p_id, bbox_inches='tight')
    
        shap_df = pd.DataFrame(shap_values_val, columns=features)
        #shap_df.to_csv(save_dir + '/evidence_%s.csv' % p_id, index=False)

    #y_pred_df.to_csv(save_dir + '/predicted_risk.csv')
    
    return shap_df, y_pred_norm, shap_plot,  data_imputed[CAT_FEATURES + CONT_FEATURES]
   
