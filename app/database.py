import random
import numpy as np
import skimage as sk
from skimage import transform
from skimage import exposure
from skimage.util import random_noise
from scipy import ndimage
import os

from app.face_recognition import *
from app.inception_blocks import *

################# IMAGE TRANSFORMATION FUNCTIONs ####################
def rotation_randomly(image):
    random_degree = random.uniform(-10,10)
    new_img = 255*transform.rotate(image,random_degree)
    new_img = new_img.astype(np.uint8)
    return new_img

def retourner_horizontal(image, prob=1.):
    random_prob = random.uniform(0.,1.)
    if random_prob < prob:
        new_img = np.fliplr(image)
    return new_img

def shear_randomly(image):
    random_degree = random.uniform(-0.1,0.1)
    afine_tf = transform.AffineTransform(shear=random_degree)
    new_img = 255*transform.warp(image, inverse_map=afine_tf)
    new_img = new_img.astype(np.uint8)
    return new_img

def change_contrast(image, percent_change=(0,15)):
    percent_change = random.uniform(percent_change[0], percent_change[1])
    v_min, v_max = np.percentile(image, (0.+percent_change, 100.-percent_change))
    new_img = exposure.rescale_intensity(image, in_range=(v_min,v_max))
    new_img = new_img.astype(np.uint8)
    return new_img

def correction(image, gamma_range=(0.7,1.0)):
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    new_img = exposure.adjust_gamma(image, gamma=gamma, gain=random.uniform(0.8,1.0))
    new_img = new_img.astype(np.uint8)
    return new_img

def blur_image(image):
    new_img = ndimage.uniform_filter(image, size=(5, 5, 1))
    new_img = new_img.astype(np.uint8)
    return new_img

def ajouter_noise(image):
    new_img = 255*random_noise(image)
    new_img = new_img.astype(np.uint8)
    return new_img

################### APPLY AUGMENTATION ######################
avail_transforms = {'rotate': rotation_randomly,
                    'shear': shear_randomly,
                    'contrast': change_contrast,
                    'gamma': correction}
                  
def apply_transform(image, num_transform=2):
    '''
    Apply random transformation to the input image
    
    Arguments:
    ---------
        image : array, uint8
            input image
        num_transform : int
            number of desired transformation
            
    Returns:
    --------
        output_image:
            output image
            
    '''
    choices = random.sample(range(0,len(avail_transforms)),num_transform)
    output_image = image
    for choice in choices:
        operation = list(avail_transforms)[choice]
        output_image = avail_transforms[operation](output_image)
    return output_image

def generate_database(dirpath, modele_OpenFace, augmentations=9, output_name='database.npy'):
    '''
    Generate the database to store all the encoding for the face recognition model
    
    Arguments:
    --------
        dirpath:
            directories contains all the images for database
        modele_OpenFace:
            the face recognition to generate encoding
        augmentations:
            number of augmentation sample
        ouput_name:
            desired name of the database
                
    '''
    encoded_database = {}
    for root, dirs, files in os.walk(dirpath):
        for name in files:
            target_name = name.split('.')[0]
            file_path = str(root) + '/' + str(name)
            image = importer_image(file_path)
            operations = augmentations+1
            for i in range(operations):
                this_name = target_name + '-' + str(i)
                if i>0:
                    image = apply_transform(image, num_transform=2)
                faces, face_pos, image_with_faces = detect_faces(image)
                if not faces:
                    pass
                else:
                    face = faces[0]
                
                face_encoding = image_encoding(image, modele_OpenFace)
                encoded_database[this_name] = face_encoding
                
    np.save(output_name, encoded_database)
    


def generate_database_person(root,name, modele_OpenFace, augmentations=9):
    encoded_database = {}
    target_name = name.split('.')[0].replace(" ","")
    image = importer_image(root)
    operations = augmentations+1
    for i in range(operations):
        this_name = target_name + '-' + str(i)
        if i>0:
            image = apply_transform(image, num_transform=2)
        faces, face_pos, image_with_faces = detect_faces(image)
        if not faces:
            pass
        else:
            face = faces[0]
                 
        face_encoding = image_encoding(image, modele_OpenFace)
        encoded_database[this_name] = face_encoding
                
    return encoded_database

def generate_database_for_dict(dirpath, modele_OpenFace, augmentations=9):
   
    encoded_database = {}
    for root, dirs, files in os.walk(dirpath):
        for name in files:
            target_name = name.split('.')[0]
            file_path = str(root) + '/' + str(name)
            image = importer_image(file_path)
            operations = augmentations+1
            for i in range(operations):
                this_name = target_name + '-' + str(i)
                if i>0:
                    image = apply_transform(image, num_transform=2)
                faces, face_pos, image_with_faces = detect_faces(image)
                if not faces:
                    pass
                else:
                    face = faces[0]
                
                face_encoding = image_encoding(image, modele_OpenFace)
                encoded_database[this_name] = face_encoding
                
    return encoded_database

