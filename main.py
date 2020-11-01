import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf

import keras
from keras.preprocessing.image import ImageDataGenerator


from PIL import Image
from recorte import Recortar
import os

BS = 128

""" Esta función genera los datos necesarios para pasarlos através de la red neuronal profunda"""
def generatedata(dataset, aug=False):
    if aug:
        datagen = ImageDataGenerator(
                            rescale=1./255,
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
            dataset,
            target_size=(48, 48),
            color_mode='grayscale',
            shuffle = True,
            class_mode='categorical',
            batch_size=BS)


"""Se debe introducir la ruta del modelo guardado al cargarlo"""
new_model = tf.keras.models.load_model(r'C:\Users\Hugo\Desktop\FacialRecog\models\lastestfacial.h5')

recortar = Recortar()

"""Localización de las fotos para recortar y para guardar"""
path_fotos_recortar = r'C:\Users\Hugo\Desktop\FacialRecog\faces\fotos'
path_guardar_fotos_recortadas = r'C:\Users\Hugo\Desktop\FacialRecog\faces\recortadas\finales'

for n in os.listdir(path_fotos_recortar):

    try:
        print(n)
        array = recortar.cortar(path_fotos_recortar + '/' + str(n))
    except:
        print('Error al cargar ' + str(n) )

    imagenes_recortadas = Image.fromarray(array)
    imagenes_recortadas.save(path_guardar_fotos_recortadas + '/' + str(n))

datos_imagenes = generatedata('faces/recortadas')

end_result = new_model.predict(datos_imagenes)

dic = {0:'Angry', 1:'Disgusted', 2:'Fearful', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surpised'}

m = 0
nombre_fotos = []
for n in os.listdir(path_guardar_fotos_recortadas):
    nombre_fotos.append(str(n))
    
for n in end_result:
    print('Foto: ' + nombre_fotos[m])
    print('--------------')
    l = 0
    for n in n: 
        print(dic[l] + ' Prob.: ' + str(n))
        l += 1
    print('--------------')
    m +=1