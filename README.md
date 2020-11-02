# Deteccion-de-emociones-a-partir-de-imagenes-faciales

Modelización de Problemas de Empresa Problema Propuesto

# Descripción:

La visión computacional es una de las áreas que ha avanzado más rápidamente enlos últimos  añosgracias  al  Aprendizaje  Profundo,  incorporándose  estas  soluciones  cada  vez  más  en nuestra vida cotidiana. Motivado por ello, se plantea el siguiente problema:Las  personas  tenemos  la  capacidad  de  reconocer  y  distinguir  rostros,  así  como  de  detectar  el sentimiento  de  las  personas  a  partir  de  su  expresión  facial.  Hoy  en  día,  los  ordenadores  también disponen de esta capacidad gracias a los avances de esta disciplina.En  esta  línea,  el  objetivo  de  este  ejercicio  es  desarrollar  un  modelo  que  sea  capaz  de  detectar  la emoción de las personas a partir de su expresión facial, permitiendo a los participantes sumergirse en el campo de la visión computacional de la mano de un problema real.

# Solución:
Base de imágenes: https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view

En primer lugar, se crea la red neuronal profunda donde el input son imagenes de 48X48 en tonos grises y el resultado siete diferentes estados de ánimo. Después se entrena la red neuronal profunda guardando los párametros de la red neuronal si se alcanza un valor de exactitud mayor al del paso anterior. En 150 epochs alcanza un valor de exactitud de 0.66 aproximadamente. El código usado se encuentra en el archivo FacialRecognition.ipynb

Para probar el modelo sobre otras imagenes se crea recorte.py, que nos permite identificar caras en fotos y recortarlas. En el archivo main.py se reproduce el proceso de recorte de las caras de las imagenes, que se guardan a parte, y a cada foto se le asigna una probabilidad de cada una de las 7 emociones establecidas. En la consola se imprime el nombre de la foto y los valores de las probabilidades asociadas.

En el archivo FacialRecognition.ipynb se puede ver un ejemplo de uso y comparar con las imagenes en la carpeta faces.

Pipeline:

![alt text](https://raw.githubusercontent.com/hdelcast/Deteccion-de-emociones-a-partir-de-imagenes-faciales/main/Pipeline.png)



# Instrucciones de uso:

como usarlo
