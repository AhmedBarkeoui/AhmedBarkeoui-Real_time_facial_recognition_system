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
from core.settings import DATABASE_IMG,NN4_SMALL2,APP,DATABASE_DIR,FILES


modele_OpenFace = faceRecoModel(input_shape=(3,96,96))
modele_OpenFace.load_weights(NN4_SMALL2)
face_dictionnaire = np.load(DATABASE_IMG, allow_pickle= True ).item()

pred = set()
liste_person = []

for (name, encodes) in face_dictionnaire.items():
    label = name
    label = ''.join([i for i in name if not i.isdigit()])
    label = label.replace('_',' ')
    label = label.replace('-','')
    #print(label)
    liste_person.append(label)
liste_person = set(liste_person)


dict_range = {10:50 ,20:40 ,30:20 ,40:15 ,50:10,60:9,70:8 ,80:5 ,90:1 ,100:0}

@login_required(login_url="/login/")
def index(request):
 
    context = {}
    context['segment'] = 'index'

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
        if load_template=='page-blank.html':
            context['liste_person'] = ','.join(liste_person)
        elif load_template == 'ui-tables.html':
            context['users'] = ','.join(liste_person) 
            context['DATABASE_DIR'] = DATABASE_DIR
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
    

def media(request): 
    media = request.POST.get('pathname')
    Person = request.POST.get('PersonName') 
    rang = request.POST.get('range')
    num_image = request.POST.get('numImage')
    Time = request.POST.get('time')


    print("num_image",num_image)
    person = ""
    ll = FILES+media
    image = cv2.imread(ll)
    if request.POST.get('imagetext') :
        new_person_image = request.POST.get('imagetext') 
        new_person_name  = new_person_image[0:-4]
    else :
        new_person_image = ""
        new_person_name = ""
    
    liste =[]
    cap = cv2.VideoCapture(FILES+media)
    time_start = time.time()
    target = dict_range[int(rang)]
    print("target",target)
    counter = 0  
    list_img=[]
    list_name=[]
    dict_name={}
    dict_time={}
    msg=None 
    size = 0
    if new_person_image :
        person = new_person_name.replace("_"," ")
        print(person)
        output = generate_database_person(FILES+new_person_name+'.jpg',new_person_name+'.jpg',modele_OpenFace, augmentations=3)
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
                var_decode = u'data:img/jpg;base64,'+data64.decode('utf-8')
                dict_name[var_decode] = [*face_out_image[1].keys()]
                liste = face_out_image[1].keys()
                msg=None
            else :
                msg=" does not appear in this video" 
        elif len(face_out_image[1].keys())>0 : 
            var_decode = u'data:img/jpg;base64,'+data64.decode('utf-8')
            dict_name[var_decode] = [*face_out_image[1].keys()]
            liste = face_out_image[1].keys()
            msg=None        
        else:
            msg="This video does not contain any identified person." 
        
        
    else:
    
        while(cap.isOpened()):  
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
                        #print("---------------------------",int(hours),":",int(minutes),":",int(seconds))
                        list_img = u'data:img/jpg;base64,'+data64.decode('utf-8')
                        list_name = [*face_out[1].keys()]
                        dict_name[list_img] = list_name
                        if Time == "on" :
                            dict_time[list_img] = str(int(hours))+":"+str(int(minutes))+":"+str(int(seconds))
                        else:
                            pass
                        msg=None
                    else :
                        msg=" does not appear in this video" 

                elif len(face_out[1].keys())>0 : 
                    list_img = u'data:img/jpg;base64,'+data64.decode('utf-8')
                    list_name = [*face_out[1].keys()]
                    dict_name[list_img] = list_name
                    if Time == "on" :
                        dict_time[list_img] = str(int(hours))+":"+str(int(minutes))+":"+str(int(seconds))
                    else:
                        pass
                    msg=None        
                else:
                    msg="This video does not contain any identified person." 
            else:      
                    ret = cap.grab()      
                    counter += 1 
            if (num_image=='one') and (person in liste):
                break
                
    print(person in liste)
    print(num_image)
                
    cap.release()
    cv2.destroyAllWindows() 
    myset = set(liste)
    if "Unknown" in myset:
        myset.remove("Unknown") 
    else:
        pass 
    time_end = time.time() 
    
    print('Total run time: %.2f s' %((time_end-time_start))) 
    size = len(myset)
    return render(request, 'page-blank.html',{"dict_name":dict_name,"dict_time":dict_time,"msg":msg,"Person":person,"len":size,"liste_person":','.join(liste_person)})


def video_names(request):
    mylist = ','.join([str(i) for i in pred])     
    return HttpResponse(mylist)    
    
def delete_from_db(request): 
    try: 
        type_del = request.GET.get('type')
        name = request.GET.get('person_name')
        print('--------------------'+name.replace(' ','_')+"------------")  

        for key in list(face_dictionnaire.keys()):   
            if name in key:
                del face_dictionnaire[key]
                np.save(DATABASE_IMG, face_dictionnaire) 
        if type_del== 'all_image':
            if name.replace('_',' ') in liste_person:
                liste_person.remove(name.replace('_',' ')) 
                shutil.rmtree(DATABASE_DIR+name.replace(' ','_')[0:-1])
        else:
            os.remove(DATABASE_DIR+name[0:-4]+"\\"+name+".jpg")
        data = {'name':name}
        stat = 200
    except:
        stat = 400
        data = {'error':'error'}
    return JsonResponse(data, status=stat)




def check_length(request):
    name = request.GET.get('check_length') 
    img_list = os.listdir(DATABASE_DIR+name) 
    return HttpResponse(len(img_list))
 
def add_single_image(request):
    try:
        person = request.GET.get('person')
        image = request.GET.get('image')
        img_list = os.listdir(DATABASE_DIR+person)
        image_transform = importer_image(FILES+image)
        number = len(img_list)+1
        image_name = person+'_00'+str(number)+'.'+image.split('.')[1]
        new_path = DATABASE_DIR+person+"\\"+image_name
        shutil.copyfile(FILES+image, new_path)

        output = generate_database_person(DATABASE_DIR+person+"\\"+image_name,image_name,modele_OpenFace, augmentations=3)
        face_dictionnaire[str(list(output.keys())[0])] = list(face_dictionnaire.values())[0]
        np.save(DATABASE_IMG, face_dictionnaire)
        data = {'number':number,'ext':image.split('.')[1],'id':image_name}
        stat = 200
    except:
        stat = 400
        data = {'error':'error'}
    return JsonResponse(data, status=stat)
      

def add_to_database(request):
    try:
        path = request.GET.get('path').replace(' ','_')
        img_path = request.GET.get('img_path')
        new_path = DATABASE_DIR+path+"\\"+img_path

        if not os.path.exists(DATABASE_DIR+path):
            os.makedirs(DATABASE_DIR+path)
            shutil.copyfile(FILES+img_path, new_path)
            os.rename(new_path,DATABASE_DIR+path+"\\"+path+"_001"+"."+img_path.split('.')[1])


        image_transform = importer_image(DATABASE_DIR+path+"\\"+path+"_001"+"."+img_path.split('.')[1])

        new_dict = generate_database_for_dict(DATABASE_DIR+path+"\\", modele_OpenFace, augmentations=3) 
        face_dictionnaire.update({key:value for key, value in new_dict.items()})
        np.save(DATABASE_IMG, face_dictionnaire)
        path = path.replace('_',' ')
        liste_person.add(path+" ")
        data = {'path':path}
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
