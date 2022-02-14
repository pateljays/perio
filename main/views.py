from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from main.models import *
from main.predict_tool.prediction_tool import predict_tool
from django_pandas.io import read_frame
from django.conf import settings
import os, json

import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import pickle

f_data_dict = {'ASA': 'American Society of Anesthesiologists',
                'DMFT': 'Calculation of Decayed, Missing, Filled index for number of teeth',
                'DMFS': 'Calculation of Decayed, Missing, Filled index for number of teeth',
                'Teeth': 'Number of teeth present',
                'Med1': 'CARDIOLOGY category',
                'Med2': 'NEPHROLOGY category',
                'Med3': 'NEUROLOGY category',
                'Med4': 'HEMATOLOGY / ONCOLOGY category',
                'Med5': 'ENDOCRINOLOGY category',
                'Med6': 'RHEUMATOLOGY category',
                'Med7': 'PULMONARY category',
                'Med8': 'GASTROENTEROLOGY category',
                'Med9': 'INFECTIOUS DISEASES category',
                'Med10': 'History of Cancer CATEGORY',
                'Med11': 'Bisphosponates/Osteoporosis',
                'TeethLost': 'Number of teeth lost due to Perio disease',
                'Smoking': 'Current smoking status',
                'Diabetes': 'Diabetes and HbA1c control',
                'PlaqueIdx': 'Plaque Index: % of sites with plaque',
                'BOP': 'Bleeding on Probing: % of sites bleeding',
                'BoneLoss': 'Any Boneloss',
                'BL_URQ': 'BoneLoss level in UR Quad',
                'BL_LRQ': 'BoneLoss level in UR Quad',
                'BL_ULQ': 'BoneLoss level in UR Quad',
                'BL_LLQ': 'BoneLoss level in UR Quad',
                'BL_Vert': 'Any Vertical Boneloss',
                'BL_MaxPct': 'Maximum boneloss % in any quad',
                'Calc_URQ': 'Subgingival Calculus Upper Right Quadrant',
                'Calc_LRQ': 'Subgingival Calculus Lower Right Quadrant',
                'Calc_ULQ': 'Subgingival Calculus Upper Left Quadrant',
                'Calc_LLQ': 'Subgingival Calculus Lower Left Quadrant',
                'RiskScore': 'Perio Risk score (as calculated)',
                'RiskSel': 'Perio Risk (as selected by provider)',
                'DxSel': 'Perio Diagnosis (as selected by provider)',
                'DxCal': 'Perio Diagnosis (as calculated by 2017 algorithm)',
                'DxCal_BL': "Perio Diagnosis (calc'd by 2017 algorithm); Boneloss is requirement for Periodontitis dx",
                'Prognosis': 'Perio Prognosis (as selected by provider)',
                'RFG1': 'General Risk Factor: Inadequate home plaque control',
                'RFG2': 'General Risk Factor: Inadequate pt compliance',
                'RFG3': 'General Risk Factor: Smoking habit',
                'RFG4': 'General Risk Factor: High level of stress',
                'RFG5': 'General Risk Factor: Bruxism/parafunctional habit',
                'RFG6': 'General Risk Factor: Diabetes mellitus',
                'RFG7': 'General Risk Factor: Other systemic medical condition',
                'RFG8': 'General Risk Factor: Medications affecting periodontium',
                'RFL1': 'Local Risk Factor: Perio attachment loss',
                'RFL2': 'Local Risk Factor: Increased probing depths',
                'RFL3': 'Local Risk Factor: Tooth mobility',
                'RFL4': 'Local Risk Factor: Furcation involvement',
                'RFL5': 'Local Risk Factor: High caries activity',
                'RFL6': 'Local Risk Factor: Defective restorations',
                'RFL7': 'Local Risk Factor: Removable partial denture(s)',
                'RFL8': 'Local Risk Factor: Tooth crowding/root proximity/open contacts',
                'RFL9': 'Local Risk Factor: Abnormal tooth anatomy',
                'RFL10': 'Local Risk Factor: Radiographic findings',
                'SRPSxPerio': 'Dental History: Scaling/surgery for perio disease',
                'OralSx': 'Dental History: Oral surgery'}


# Create your views here.

def main_index(request):
    if request.method == 'GET':
        f_plot = {'pf_cate':[], 
                'pf_val':[], 
                'rf_cate':[],
                'rf_val':[], 
                'f_data_dict':{}}
        params = {'result_display_status':'init', 
                'p_id':'', 
                'y_pred_norm':'',
                'y_pred_norm_cate':[],
                'bullet_data':json.dumps(''),
                'f_plot':f_plot, 
                'shap_plot':''}
        return render(request, 'main/index.html', {'params':params})
    elif request.method == 'POST':
        #get user input
        try:
            p_id = int(request.POST.get('PID'))
        except:
            params = {'result_display_status':'p_id_not_right'}
            return render(request, 'main/index.html', {'params':params})
        
        #check if patient exist
        if PatientInfo.objects.filter(StudyId=p_id).exists():
            '''
            prepare patient information
            '''
            p_rec = read_frame(PatientInfo.objects.filter(StudyId=p_id))
            p_rec = p_rec.fillna(value=np.nan)
            file_dir = os.path.join(settings.BASE_DIR, 'main/predict_tool/')
            shap_df, y_pred_norm, shap_plot = predict_tool(file_dir, p_rec)
            
            #round risk score
            y_pred_norm = round(y_pred_norm[0],2)
            #assign a category
            y_pred_norm_cate_all = ['Healthy Periodontium', 'Healthy Periodontium with Attachment Loss', 
                                    'Healthy Periodontium with Attachment Loss', 'Gingivitis', 
                                    'Slight Perio (Stage 1) ', 'Moderate Perio (Stage 2) ', 
                                    'Moderate Perio (Stage 2) ', 'Severe Perio (Stage 3)', 
                                    'Severe Perio (Stage 4) ', 'Aggressive Periodontitis']
            y_pred_norm_cate =y_pred_norm_cate_all[int(y_pred_norm/10)]
            #prepare bullet data
            bullet_data = [{"title":"PD Risk Score",
                            "subtitle":"in total", 
                            "ranges":[10,20,30,40,50,60,70,80,90,100], 
                            "measures":[y_pred_norm], 
                            "markers":[y_pred_norm]}]
            #prepare protective factors plot
            pf_data = shap_df.T
            pf_data = pf_data.loc[pf_data[0]<0].sort_values(by=0)
            pf_cate = list(pf_data.index)[0:10]
            pf_val = list(pf_data.iloc[0:10,0])
            pf_val = [-i for i in pf_val]
            #prepare risk factors plot
            rf_data = shap_df.T
            rf_data = rf_data.loc[rf_data[0]>0].sort_values(by=0, ascending=False)
            rf_cate = list(rf_data.index)[0:10]
            rf_val = list(rf_data.iloc[0:10,0])
            
            
            #put data together
            f_plot = {'pf_cate':pf_cate, 
                    'pf_val':pf_val, 
                    'rf_cate':rf_cate,
                    'rf_val':rf_val, 
                    'f_data_dict':f_data_dict}
            
            see=rf_val
            params = {'result_display_status':'show_result', 
                    'p_id':p_id, 
                    'y_pred_norm':y_pred_norm,
                    'y_pred_norm_cate':y_pred_norm_cate,
                    'bullet_data':json.dumps(str(bullet_data).replace("'",'"')),
                    'f_plot':f_plot, 
                    'see':see, 
                    'shap_plot':shap_plot}
            return render(request, 'main/index.html', {'params':params})
        else:
            params = {'result_display_status':'no_result'}
            return render(request, 'main/index.html', {'params':params})


###### ajax functions #######

def get_pid(request):
    try:
        p_id = int(request.GET.get('term', ''))
    except:
        data = 'fail_1'
        mimetype = 'application/json'
        return HttpResponse(data, mimetype)
    
    p_info = PatientInfo.objects.filter(StudyId__startswith=p_id).values_list('StudyId', flat=True)
    p_info=list(set(p_info))[:10]
    p_info.sort()
    results = []
    for pid in p_info:
        p_json = {}
        p_json['value'] = pid
        results.append(p_json)
    if len(results)==0:
        results=[{'value':'Patient NOT in Database'}]
    
    data = json.dumps(results)
    mimetype = 'application/json'
    return HttpResponse(data, mimetype)















