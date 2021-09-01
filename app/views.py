 # -*- encoding: utf-8 -*-
"""  
Copyright (c) 2019 - present AppSeed.us    
""" 
 
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django.http import StreamingHttpResponse
from django import template
from app.Camera import VideoCamera
from app.Camera import IPWebCam
from django.contrib.auth.models import User
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import json
import time
import cv2
import numpy as np
import collections
from PIL import Image
from app.face_recognition import *
from app.database import *
from app.alignement import AlignDlib
from app.inception_blocks import *
import io
from io import BytesIO
import urllib, base64
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from django.http import JsonResponse
import shutil
import os
from django.template.defaulttags import register
from core.settings import DATABASE_IMG,NN4_SMALL2,APP,DATABASE_DIR,FILES,DATABASE_DATE_ADDED,HISTORY,DATABASE_SIGLE_IMG_DIR,DATASET
from datetime import datetime
from django.core.serializers.json import DjangoJSONEncoder
import pandas as pd
import numpy as np



modele_OpenFace = faceRecoModel(input_shape=(3,96,96))
modele_OpenFace.load_weights(NN4_SMALL2)
face_dictionnaire = np.load(DATABASE_IMG, allow_pickle= True ).item()
face_dictionnaire = collections.OrderedDict(sorted(face_dictionnaire.items()))

Date_Added =  np.load(DATABASE_DATE_ADDED, allow_pickle= True ).item()
History = pd.read_csv(HISTORY ,index_col=0)
Dataset = pd.read_csv(DATASET ,index_col=0)

pred = set() 
liste_person = []
mylist = {}

for (name, encodes) in face_dictionnaire.items():
    label = name
    label = ''.join([i for i in name if not i.isdigit()])
    label = label.replace('_',' ')
    label = label.replace('-','')
    liste_person.append(label)
liste_person = set(liste_person)   

 
dict_range = {10:50 ,20:40 ,30:20 ,40:15 ,50:10,60:9,70:8 ,80:5 ,90:1 ,100:0}

@login_required(login_url="/login/")
def index(request):
 
    context = {}
    context['segment'] = 'index' 
    context['History'] = History.set_index('Person').T.to_dict('list')
    html_template = loader.get_template( 'index.html' )
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def pages(request):  
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    
    try:
        load_template      = request.path.split('/')[-1]
        context['segment'] = load_template
        if load_template == 'Advanced-video-processing.html':
            context['liste_person'] = ','.join(liste_person)
        elif load_template == 'Database-Management.html':
            dataJSON = json.dumps(Date_Added, indent=4, sort_keys=True, default=str) 
            context['users'] = ','.join(liste_person) 
            context['DATABASE_DIR'] = DATABASE_DIR
            context['Date_Added'] = dataJSON
        elif load_template == 'charts-morris.html':
            context['Dataset'] = Dataset
        else:
            pass
        html_template = loader.get_template( load_template )
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template( 'page-404.html' )
        return HttpResponse(html_template.render(context, request))

    except:

        html_template = loader.get_template( 'page-500.html' )
        return HttpResponse(html_template.render(context, request))

def gen(camera):
    while True:
        frame = camera.get_frame()[0]
        pred.update(camera.get_frame()[1])
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed(request): 
    return StreamingHttpResponse(gen(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame')

def webcam_feed(request):
    return StreamingHttpResponse(gen(IPWebCam()),
                    content_type='multipart/x-mixed-replace; boundary=frame')
    

def Advanced_video_processing(request): 
    try:
        media = request.POST.get('pathname')
        Person = request.POST.get('PersonName') 
        rang = request.POST.get('range')
        num_image = request.POST.get('numImage')
        Time = request.POST.get('time')
        time_start = request.POST.get('time_start')
        time_end = request.POST.get('time_end')
        intervall_start = 'none'
        intervall_end = 'none'
        person = ""
        ll = FILES+media
        image = cv2.imread(ll)
        if request.POST.get('imagetext') :
            new_person_image = request.POST.get('imagetext') 
            new_person_name  = request.POST.get('nametext') 
        else :
            new_person_image = ""
            new_person_name = ""

        liste =[]
        cap = cv2.VideoCapture(FILES+media)
        fps_cap = cap.get(cv2. CAP_PROP_FPS)
        frame_count_cap = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
        check_cap = 1000 * int(frame_count_cap/fps_cap)+1000
        
        if time_end:
            time_start_miliseconds = int(3600000 * int(time_start.split(":")[0]) + 60000 * int(time_start.split(":")[1]) + 1000 * int(time_start.split(":")[2]))
            time_end_miliseconds = int(3600000 * int(time_end.split(":")[0]) + 60000 * int(time_end.split(":")[1]) + 1000 * int(time_end.split(":")[2]))
            if (time_end_miliseconds-time_start_miliseconds)!=check_cap:
                intervall_start = time_start
                intervall_end = time_end
            else:
                intervall_start = 'none'
                intervall_end = 'none'

        target = dict_range[int(rang)]
        counter = 0  
        list_img=[]
        list_name=[]
        dict_name={}
        dict_time={}
        msg=None 
        size = 0 
        if new_person_image :
            person = new_person_name.replace("_"," ")
            a = ""
            while a != 'ok':
                try:
                    output = generate_database_person(FILES+new_person_image,new_person_name.replace(" ","_"),modele_OpenFace, augmentations=3)
                    a = 'ok'
                except:
                    a = 'notok'
        elif Person:
            person = Person
            output = face_dictionnaire
        else:
            output = face_dictionnaire


        ext = media.split('.')[1]
        list_ext = ["jpeg","jpg","png","gif","tif","psd","pdf","eps","ai","indd","svg"]
        if(ext in list_ext):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_out_image = face_recognition_image(gray, modele_OpenFace,face_dictionnaire , plot=True, faces_out=True)
            im = Image.fromarray(face_out_image[0])
            data = BytesIO()        
            im.save(data, "JPEG")        
            data64 = base64.b64encode(data.getvalue()) 

            if person or new_person_image:
                if person in face_out_image[1].keys(): 
                    var_decode = u'data:img/'+ext+';base64,'+data64.decode('utf-8')
                    dict_name[var_decode] = [*face_out_image[1].keys()]
                    liste = face_out_image[1].keys()
                    msg=None
                else :
                    msg=" does not appear in this image" 
            elif len(face_out_image[1].keys())>0 : 
                var_decode = u'data:img/'+ext+';base64,'+data64.decode('utf-8')
                dict_name[var_decode] = [*face_out_image[1].keys()]
                liste = face_out_image[1].keys()
                msg=None        
            else:
                msg="This image does not contain any identified person." 


        else:
            cap.set(cv2.CAP_PROP_POS_MSEC,time_start_miliseconds)
            while cap.get(cv2.CAP_PROP_POS_MSEC) <= time_end_miliseconds:
                if counter == target:      
                    flag,frame = cap.read()      
                    if flag == False:    
                        break 
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #frame_flip = cv2.flip(gray, -1)
                    milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
                    seconds = milliseconds//1000
                    milliseconds = milliseconds%1000
                    minutes = 0
                    hours = 0
                    if seconds >= 60:
                        minutes = seconds//60
                        seconds = seconds % 60

                    if minutes >= 60:
                        hours = minutes//60
                        minutes = minutes % 60


                    face_out = face_recognition_image(gray, modele_OpenFace, output, plot=True, faces_out=True)

                    im = Image.fromarray(face_out[0])
                    data = BytesIO()        
                    im.save(data, "JPEG")        
                    data64 = base64.b64encode(data.getvalue()) 
                    liste+=face_out[1].keys()
                    counter = 0 

                    if person or new_person_image:
                        if person in face_out[1].keys():
                            list_img = u'data:img/'+ext+';base64,'+data64.decode('utf-8')
                            list_name = [*face_out[1].keys()]
                            dict_name[list_img] = list_name
                            if Time == "on" :
                                dict_time[list_img] = str(int(hours)).zfill(2)+":"+str(int(minutes)).zfill(2)+":"+str(int(seconds)).zfill(2)
                            else:
                                pass
                            msg=None
                        else :
                            msg=" does not appear in this video" 

                    elif len(face_out[1].keys())>0 : 
                        list_img = u'data:img/'+ext+';base64,'+data64.decode('utf-8')
                        list_name = [*face_out[1].keys()]
                        dict_name[list_img] = list_name
                        if Time == "on" :
                            dict_time[list_img] = str(int(hours)).zfill(2)+":"+str(int(minutes)).zfill(2)+":"+str(int(seconds)).zfill(2)
                        else:
                            pass
                        msg=None        
                    else:
                        msg="This video does not contain any identified person" 
                else:      
                        ret = cap.grab()      
                        counter += 1 
                if (num_image=='one') and (person in liste):
                    break


        cap.release()
        cv2.destroyAllWindows() 
        myset = set(liste)
        if "Unknown" in myset:
            myset.remove("Unknown") 
        else:
            pass 
        extension = "image" if ext in ["jpeg","jpg","png","gif","tif","psd","pdf","eps","ai","indd","svg"] else "video"
        size = len(myset)    
         
        return render(request, 'Advanced-video-processing.html', {"intervall_start":intervall_start,"intervall_end":intervall_end,"dict_name":dict_name,"dict_time":dict_time,"msg":msg,"extension":extension,"Person":person,"len":size,"liste_person":','.join(liste_person)})
    except : 
        return render(request, 'Advanced-video-processing.html',{"liste_person":','.join(liste_person)})

def video_names(request):
    historyy = History
    mylist = ','.join([str(i) for i in pred])
    date_string = str(datetime.now().strftime("%d/%m/%Y"))
    time_string = str(datetime.now().strftime("%H:%M:%S"))
    if len(mylist) > 0 :
        for i in mylist.split(","): 
            if i not in list(historyy["Person"].unique()):
                historyy = historyy.append(dict(zip(historyy.columns,[i,date_string,time_string])), ignore_index=True)
                historyy.to_csv(HISTORY)
            else:
                result_date = list(History[History["Person"]== i]['Date'])
                if str(datetime.now().strftime("%d/%m/%Y")) not in [i[0:10] for i in result_date] :
                    historyy = historyy.append(dict(zip(historyy.columns,[i,date_string,time_string])), ignore_index=True)
                    historyy.to_csv(HISTORY)
    return HttpResponse(mylist)
     
def delete_from_db(request):    
    try: 
        type_del = request.GET.get('type')       
        name = request.GET.get('person_name')
        
        for key in list(face_dictionnaire.keys()):   
            if name in key:
                del face_dictionnaire[key]
                 
        
        if name.replace('_',' ') in Date_Added.keys():
            del Date_Added[name.replace('_',' ')]
            np.save(DATABASE_DATE_ADDED, Date_Added)
            
        if name.replace('_',' ') in liste_person:
            liste_person.remove(name.replace('_',' ')) 
            shutil.rmtree(DATABASE_DIR+name.replace(' ','_')[0:-1])
            face_dictionnaire_ordred = collections.OrderedDict(sorted(face_dictionnaire.items()))
            np.save(DATABASE_IMG, face_dictionnaire_ordred)
            
        data = {'name':name.replace('_',' ')[0:-1]}
        stat = 200
    except:
        stat = 400
        data = {'error':'error'}
    return JsonResponse(data, status=stat)


def delete_single_image(request):  
    try:   
        name = request.GET.get('person_name')
        for key in list(face_dictionnaire.keys()):   
            if name in key:
                del face_dictionnaire[key]

        if os.path.exists(DATABASE_DIR+name[0:-4]+"\\"+name+"."+request.GET.get('ext')):
            os.remove(DATABASE_DIR+name[0:-4]+"\\"+name+"."+request.GET.get('ext'))
            face_dictionnaire_ordred = collections.OrderedDict(sorted(face_dictionnaire.items()))
            np.save(DATABASE_IMG, face_dictionnaire_ordred) 
        
        data = {'name':name.replace('_',' ')[0:-1]}
        stat = 200
    except:
        stat = 400
        data = {'error':'error'}
    return JsonResponse(data, status=stat)


def check_length(request):
    name = request.GET.get('check_length') 
    img_list = os.listdir(DATABASE_DIR+name) 
    return HttpResponse(len(img_list))



def video_length(request):
    cap = cv2. VideoCapture(FILES+request.GET.get('label')) 
    fps = cap. get(cv2. CAP_PROP_FPS)
    frame_count = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    seconds = int(frame_count/fps)+1
    minutes = 0
    hours = 0
    if seconds >= 60:
        minutes = seconds//60
        seconds = seconds % 60
    
    if minutes >= 60:
        hours = minutes//60
        minutes = minutes % 60 
        str(hours).zfill(2)
    duration = str(hours).zfill(2)+':'+str(minutes).zfill(2)+':'+str(seconds).zfill(2)
    return HttpResponse(duration)
 
def add_single_image(request):
    try:
        person = request.GET.get('person')
        image = request.GET.get('image')
        img_list = os.listdir(DATABASE_DIR+person)
        image_transform = importer_image(FILES+image)
        i=1
        while os.path.exists(DATABASE_DIR+person+"\\"+person+'_00'+str(i)+'.'+image.split('.')[1]):
            i+=1
        image_name = person+'_00'+str(i)+'.'+image.split('.')[1]
        new_path = DATABASE_DIR+person+"\\"+image_name
        shutil.copyfile(FILES+image, new_path)
        a = ""
        while a != 'ok':
            try:
                output = generate_database_person(DATABASE_DIR+person+"\\"+image_name,image_name,modele_OpenFace, augmentations=3)
                a = 'ok'
            except:
                a = 'notok'
        for key,value in output.items():   
            face_dictionnaire[key] = value
        face_dictionnaire_ordred = collections.OrderedDict(sorted(face_dictionnaire.items()))
        np.save(DATABASE_IMG, face_dictionnaire_ordred)
        data = {'number':i,'ext':image.split('.')[1],'id':image_name}
        stat = 200
    except:
        stat = 400
        data = {'error':'error'}
        
    return JsonResponse(data, status=stat)
      

def add_to_database(request):   
    try:
        path = request.GET.get('path').replace(' ','_')
        img_list = request.GET.get('img_list').split(",")
        j =1
        for i in img_list:
            new_path = DATABASE_DIR+path+"\\"+i
            if not os.path.exists(DATABASE_DIR+path):
                os.makedirs(DATABASE_DIR+path) 
            shutil.copyfile(FILES+i, new_path)
            os.rename(new_path,DATABASE_DIR+path+"\\"+path+"_00"+str(j)+"."+i.split('.')[1])    
            j+=1
        db_single_img = DATABASE_SIGLE_IMG_DIR+img_list[0]
        if not os.path.exists(DATABASE_SIGLE_IMG_DIR+path.replace('_',' ')+" ."+img_list[0].split('.')[1]):
            shutil.copyfile(FILES+img_list[0], db_single_img)
            os.rename(db_single_img,DATABASE_SIGLE_IMG_DIR+path.replace('_',' ')+" ."+img_list[0].split('.')[1])
        a = ""
        while a != 'ok':
            try:
                new_dict = generate_database_for_dict(DATABASE_DIR+path+"\\", modele_OpenFace, augmentations=3) 
                a = 'ok'
            except:
                a = 'notok'
        face_dictionnaire.update({key:value for key, value in new_dict.items()})
        face_dictionnaire_ordred = collections.OrderedDict(sorted(face_dictionnaire.items()))
        np.save(DATABASE_IMG, face_dictionnaire_ordred)
        path = path.replace('_',' ')
        liste_person.add(path+" ")
        date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        Date_Added.update({path+' ': date})
        np.save(DATABASE_DATE_ADDED, Date_Added)
        data = {'path':path,'date':date}
        stat = 200
    except:
        stat = 400
        data = {'error':'error'}
    return JsonResponse(data, status=stat)



def check_image(request):
    try:
        image_path = (FILES+request.GET.get('inputValue'))
        img = cv2.imread(image_path)
        import_img = importer_image(image_path)
        face_out = face_recognition_image(import_img, modele_OpenFace, face_dictionnaire, plot=True, faces_out=True) 

        num = get_num_from_image(img)
        if num == 1:
            if 'Unknown' in list(face_out[1].keys()):
                data = {'response':'Verified',"inlist":""} 
            else:
                data = {'response':'Verified','inlist':str(list(face_out[1].keys())[0])}
        else:
            data = {'response':num,"inlist":""}
        stat = 200

    except:
        stat = 400
        data = {'error':'error'}
    return JsonResponse(data, status=stat)

def check_imagee(request):
    try:
        stat = 200
        image_path = FILES+request.GET.get('inputValue')
        img = cv2.imread(image_path)
        import_img = importer_image(image_path)
        num = get_num_from_image(img) 
        if num != 1:
            data = {'response':num,"inlist":"","duplicated":""}
        elif request.GET.get('img_list'):
            database = face_dictionnaire.copy() 
            for i in request.GET.get('img_list').split(","):
                if np.array_equal(np.array(Image.open(image_path)),np.array(Image.open(FILES+i))):
                    exist = "ok"
                    break
                else: 
                    exist = "not"
            i = 0
            if exist=="ok":
                data =  {'response':'',"inlist":"","duplicated":"duplicated"}
            else:
                #check = request.GET.get('img_list').split(",")[0]
                a = ""
                while a != 'ok':
                    try:
                        i=1
                        for j in request.GET.get('img_list').split(","):
                            face_outt1 = generate_database_person(FILES+j,"New_person_00"+str(i),modele_OpenFace, augmentations=3)
                            i+=1
                            for key,value in face_outt1.items():   
                                database[key] = value
                        a = 'ok'
                    except:
                        a = 'notok'
                for key,value in face_outt1.items():   
                    database[key] = value

                face_outt2 = face_recognition_image(import_img, modele_OpenFace, database, plot=True, faces_out=True)
                if 'Unknown' in list(face_outt2[1].keys()):
                    data = {'response':'',"inlist":"","duplicated":"notVerified"}
                else:
                    if "New person " == list(face_outt2[1].keys())[0]:
                        data = {'response':'Verified',"inlist":"","duplicated":""}
                    else:
                        data = {'response':'Verified','inlist':str(list(face_outt2[1].keys())[0]),"duplicated":""}     
            database = {}
        else:
            face_out = face_recognition_image(import_img, modele_OpenFace, face_dictionnaire, plot=True, faces_out=True)
            if 'Unknown' in list(face_out[1].keys()):
                data = {'response':'Verified',"inlist":"","duplicated":""}
            else:
                data = {'response':'Verified','inlist':str(list(face_out[1].keys())[0]),"duplicated":""}

    except:
        stat = 400
        data = {'error':'error'}
    return JsonResponse(data, status=stat)

def check_single_image_verif(request):
    stat = 200
    image1 = (FILES+request.GET.get('image1'))
    image2 = (FILES+request.GET.get('image2'))
    inlist_person = request.GET.get('inlist_person').replace(' ','_')[0:-1]
    
    import_img1 = importer_image(image1)
    import_img2 = importer_image(image2)
    img = cv2.imread(image2)
    num = get_num_from_image(img)
    im_a = Image.open(image2)
    im_A = np.array(im_a)
    im_b = Image.open(image1)
    im_B = np.array(im_b) 
    if num == 1:
        if np.array_equal(im_A,im_B):
            data = {'response':'',"exist":"same","num":""}
            #houni y9ollou nafss taswira nik omek hadhi  -_-
        else:
            a = ""
            while a != 'ok':
                try:
                    dict_1 = generate_database_person(image1,"New_person_001",modele_OpenFace, augmentations=3)
                    a = 'ok'
                except:
                    a = 'notok'
            for key,value in face_dictionnaire.items():   
                if inlist_person in key:
                    dict_1[key] = value

            face_out = face_recognition_image(import_img2, modele_OpenFace, dict_1, plot=True, faces_out=True)
            if 'Unknown' in list(face_out[1].keys())[0]:
                data = {'response':'Unknown',"exist":"","num":""} 
                # houni y9oulou taswira jdida la tchabah l titi la l ahmed -_-
            else:
                if inlist_person.replace('_',' ')+" " in list(face_out[1].keys())[0]:
                    data = {'response':'surenotnew',"exist":"","num":""}
                    # houni y9oulou barra nik omek yezzi bla 9o7b mta3ek,mta3 ahmed zeda hadhi w ysaker l modal -_-  
                else:
                    # wajdi c bon , donc taw y9ollou verified w ki y7ot save yetsajel b les 2 image w jawou bahi
                    data = {'response':'Verified',"exist":"","num":""}
    else:
        data = {'response':"","exist":"","num":num}
    return JsonResponse(data, status=stat)

def recheck_image(request):
    try:
        stat = 200
        check_image = (FILES+request.GET.get('check_image'))
        inlist_person = request.GET.get('inlist_person').replace(' ','_')[0:-1]
        im = Image.open(check_image)
        im_A = np.array(im)
        for i in os.listdir(DATABASE_DIR+inlist_person):
            if np.array_equal(im_A,np.array(Image.open(DATABASE_DIR+inlist_person+"\\"+i))):
                verif = "exist" 
                break 
            else:
                verif = "not exist"

        if verif == "exist":
            data = {'response':'exist','inlist_person':inlist_person.replace('_',' ')} 
        else:
            data = {'response':'notExist','inlist_person':inlist_person.replace('_',' ')} 
    except:
        stat = 400
        data = {'error':'error'}
    return JsonResponse(data, status=stat)

def check_single_image(request):
    try:
        image_path = (FILES+request.GET.get('inputValue'))
        name = request.GET.get('name_ss')
        img = cv2.imread(image_path)
        num = get_num_from_image(img)
        im = Image.open(image_path)
        im_A = np.array(im)
        if num == 1:
            for i in os.listdir(DATABASE_DIR+name):
                if np.array_equal(im_A,np.array(Image.open(DATABASE_DIR+name+"\\"+i))):
                    verif = "ok" 
                    break 
                else:
                    verif = "exist"
            if verif == "ok":
                data = {'response':"","inlist":"","exist":"exist","num":""} 
            else:
                import_img = importer_image(image_path)
                face_out = face_recognition_image(import_img, modele_OpenFace, face_dictionnaire, plot=True, faces_out=True)
                if list(face_out[1].keys())[0][:-1]==name.replace('_',' '):
                    data = {'response':'Verified',"inlist":"","exist":"","num":""}  
                else:
                    if 'Unknown' in list(face_out[1].keys()):
                        data = {'response':'Unknown',"inlist":"","exist":"","num":""} 
                    else:
                        data = {'response':'','inlist':str(list(face_out[1].keys())[0]),"exist":"","num":""} 
        else:
            data = {'response':"","inlist":"","exist":"","num":num}
        stat = 200
    except:
        stat = 400
        data = {'error':'error'}
    return JsonResponse(data, status=stat)
 
def edit_person(request):
    try:
        name = request.GET.get('name_edit')
        listtt = []
        img_list = os.listdir(DATABASE_DIR+name)
        data = {'list_edit':','.join(img_list)}
        stat = 200
    except:
        stat = 400
        data = {'error':'error'}
    return JsonResponse(data, status=stat)

@register.filter
def get_value(dictionary, key):
    return dictionary.get(key)

@register.filter
def split(value):
    return value.split(',')

def chart_test(request): 
    responses_pie =  Results.objects.filter(date=date.today()).annotate(count=Count('result'))  
    return render(request,'mbr/chart_test.html',{'responses_pie': responses_pie})
