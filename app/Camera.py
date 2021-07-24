# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template
import imutils
import cv2
from imutils.video import VideoStream
from app.face_recognition import *
from app.database import *
from app.alignement import AlignDlib
from app.inception_blocks import *
import urllib
from core.settings import DATABASE_IMG,NN4_SMALL2,APP,DATABASE_DIR,FILES


modele_OpenFace = faceRecoModel(input_shape=(3,96,96))
modele_OpenFace.load_weights(NN4_SMALL2)




class VideoCamera(object):
    face_dictionnaire = np.load(DATABASE_IMG, allow_pickle= True ).item()
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        #self.vs = VideoStream(src=0).start()

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    
    def get_frame(self):
        liste = set()
        flag,image = self.video.read()
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_show = face_recognition_stream(frame, modele_OpenFace, database=face_dictionnaire, plot=True,faces_out=True)
        img_show_video = cv2.cvtColor(img_show[0], cv2.COLOR_BGR2RGB)
        ret, jpeg = cv2.imencode('.jpg', img_show_video)
        return jpeg.tobytes(),img_show[2]

    
    
class IPWebCam(object):
    face_dictionnaire = np.load(DATABASE_IMG, allow_pickle= True ).item()
    def __init__(self):
        self.url = 'http://192.168.137.176:8080/shot.jpg'
        #self.vs = VideoStream(src=0).start()
        
    def __del__(self):
        cv2.destroyAllWindows()

    
    def get_frame(self):
        imgResp = urllib.request.urlopen(self.url)
        imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img=cv2.imdecode(imgNp,-1)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_show = face_recognition_stream(frame, modele_OpenFace, database=face_dictionnaire, plot=True,faces_out=True)
        img_show_video = cv2.cvtColor(img_show[0], cv2.COLOR_BGR2RGB)
        ret, jpeg = cv2.imencode('.jpg', img_show_video)
        return jpeg.tobytes(),img_show[2]