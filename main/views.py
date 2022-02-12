from django.shortcuts import render
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




# Create your views here.

def main_index(request):
    if request.method == 'GET':
        params = {'result_display_status':'init'}
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
            #rf_val = ['{0:.2E}'.format(i) for i in rf_val]
            #put data together
            f_plot = {'pf_cate':pf_cate, 
                    'pf_val':pf_val, 
                    'rf_cate':rf_cate,
                    'rf_val':rf_val}
            
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





