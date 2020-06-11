# -*- coding: utf-8 -*-
"""Copia de Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13f1mKZFtnl3shK8b-ckDmoHBwnfyltmT
"""

from google.colab import files
files.upload()
from google.colab import files
files.upload()

def ConvertTxtInVector(txt,Datos):
  Results = open (txt,'r')
  Results =Results.read().split(" ")
  for i in range (0,5):
    Datos[i]=float (Results[i])
  return Datos

import matplotlib.pyplot as plt
Hilos=[1,2,4,8,16]
Datos=[-1,-1,-1,-1,-1]

plt.figure()


ConvertTxtInVector('minion.jpg_3.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVector('minion.jpg_7.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVector('minion.jpg_9.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVector('minion.jpg_15.txt',Datos)
plt.plot(Hilos,Datos)
plt.ion()
plt.xlabel('NUMERO DE HILOS', fontsize=13)
plt.ylabel('TIEMPO(segundos)', fontsize=13)
plt.title('TIEMPO DE EJECUCION PARA UNA IMAGEN EN HD(1080x720)', fontsize=17)
plt.legend(['Kernnel 3','Kernnel 5','Kernnel 7','Kernnel 15'])
plt.savefig("GraficaHD.jpg")

plt.figure()
ConvertTxtInVector('landscape.jpg_3.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVector('landscape.jpg_7.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVector('landscape.jpg_9.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVector('landscape.jpg_15.txt',Datos)
plt.plot(Hilos,Datos)
plt.ion()
plt.xlabel('NUMERO DE HILOS', fontsize=13)
plt.ylabel('TIEMPO(segundos)', fontsize=13)
plt.title('TIEMPO DE EJECUCION PARA UNA IMAGEN EN FHD(1920x1080)', fontsize=17)
plt.legend(['Kernnel 3','Kernnel 5','Kernnel 7','Kernnel 15'])
plt.savefig("GraficaFHD.jpg")
plt.figure()
ConvertTxtInVector('universe.jpg_3.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVector('universe.jpg_7.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVector('universe.jpg_9.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVector('universe.jpg_15.txt',Datos)
plt.plot(Hilos,Datos)
plt.style.use('seaborn')
plt.xlabel('NUMERO DE HILOS', fontsize=13)
plt.ylabel('TIEMPO(segundos)', fontsize=13)
plt.title('TIEMPO DE EJECUCION PARA UNA IMAGEN EN 4K(3840x2160)', fontsize=17)
plt.legend(['Kernnel 3','Kernnel 5','Kernnel 7','Kernnel 15'])
plt.savefig("Grafica4K.jpg")
#SpeedUp

def ConvertTxtInVectorSpeedUp(txt,Datos):
  Results = open (txt,'r')
  Results =Results.read().split(" ")
  for i in range (0,5):
    Datos[i]=float (Results[i])
    Datos[i]=Datos[i]*1000
  TSecuencial=Datos[0]
  for i in range (0,5):
    Datos[i]=TSecuencial/Datos[i]
  return Datos

import matplotlib.pyplot as plt
Hilos=[1,2,4,8,16]
Datos=[-1,-1,-1,-1,-1]

plt.figure()


ConvertTxtInVectorSpeedUp('minion.jpg_3.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVectorSpeedUp('minion.jpg_7.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVectorSpeedUp('minion.jpg_9.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVectorSpeedUp('minion.jpg_15.txt',Datos)
plt.plot(Hilos,Datos)
plt.ion()
plt.xlabel('NUMERO DE HILOS', fontsize=13)
plt.ylabel('SpeedUp( Sp )', fontsize=13)
plt.title('EJECUCION PARA UNA IMAGEN EN HD(1080x720)', fontsize=17)
plt.legend(['Kernnel 3','Kernnel 5','Kernnel 7','Kernnel 15'])
plt.savefig("GraficaHDSpeedUp.jpg")

plt.figure()
ConvertTxtInVectorSpeedUp('landscape.jpg_3.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVectorSpeedUp('landscape.jpg_7.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVectorSpeedUp('landscape.jpg_9.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVectorSpeedUp('landscape.jpg_15.txt',Datos)
plt.plot(Hilos,Datos)
plt.ion()
plt.xlabel('NUMERO DE HILOS', fontsize=13)
plt.ylabel('SpeedUp( Sp )', fontsize=13)
plt.title('EJECUCION PARA UNA IMAGEN EN FHD(1920x1080)', fontsize=17)
plt.legend(['Kernnel 3','Kernnel 5','Kernnel 7','Kernnel 15'])
plt.savefig("GraficaFHDSpeedUp.jpg")
plt.figure()
ConvertTxtInVectorSpeedUp('universe.jpg_3.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVectorSpeedUp('universe.jpg_7.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVectorSpeedUp('universe.jpg_9.txt',Datos)
plt.plot(Hilos,Datos)
ConvertTxtInVectorSpeedUp('universe.jpg_15.txt',Datos)
plt.plot(Hilos,Datos)
plt.style.use('seaborn')
plt.xlabel('NUMERO DE HILOS', fontsize=13)
plt.ylabel('SpeedUp( Sp)', fontsize=13)
plt.title('EJECUCION PARA UNA IMAGEN EN 4K(3840x2160)', fontsize=17)
plt.legend(['Kernnel 3','Kernnel 5','Kernnel 7','Kernnel 15'])
plt.savefig("Grafica4KSpeedUp.jpg")
#SpeedUp