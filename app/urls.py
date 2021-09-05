# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from app import views

urlpatterns = [

    # The home page
    path('', views.index, name='home'),
    
    path('video_feed', views.video_feed, name='video_feed'),
    
    path('webcam_feed', views.webcam_feed, name='webcam_feed'),
    
    path('Advanced_video_processing', views.Advanced_video_processing, name='Advanced_video_processing'),
    
    path('check_image', views.check_image, name='check_image'),
    
    path('check_imagee', views.check_imagee, name='check_imagee'),
    
    path('video_names', views.video_names, name='video_names'),
    
    path('delete_from_db', views.delete_from_db, name='delete_from_db'),
    
    path('add_to_database', views.add_to_database, name='add_to_database'),

    path('edit_person', views.edit_person, name='edit_person'),
    
    path('check_length', views.check_length, name='check_length'),
    
    path('add_single_image', views.add_single_image, name='add_single_image'),
    
    path('check_single_image', views.check_single_image, name='check_single_image'),
    
    path('recheck_image', views.recheck_image, name='recheck_image'),
    
    path('check_single_image_verif', views.check_single_image_verif, name='check_single_image_verif'),
    
    path('video_length', views.video_length, name='video_length'),
    
    path('delete_single_image', views.delete_single_image, name='delete_single_image'),
    
    path('chart_test', views.chart_test, name='chart_test'),
    
    path('find_in_db', views.find_in_db, name='find_in_db'),
    
    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),
   

]
