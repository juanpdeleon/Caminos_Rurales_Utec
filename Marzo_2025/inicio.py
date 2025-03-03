import cv2
import copy
from ultralytics import solutions
import os
import shutil
import json
from libreria import process_input_video

# Leer el archivo de configuración de los parametros 
with open("config.json", "r") as file:
    config = json.load(file)

# Acceder a los parámetros
Input_files = config["Input_files"]  # Ubicacion a donde van a estar los archivos originales
Output_files = config["Output_files"] # ubicacion a donde se guaardan las imagnes y los json
Backup_files = config["Backup_files"] # Ubicacion a donde se hace el respaldo de los archivos
Log_dir = config["Log_dir"]   #implementar las salidas del print a un archivo de log, para ver los errores ya que el procedimiento gira automatico.
#preocess_input_video

#LLamada al sistema 
# Terimnar de implementar, para los parametros recibidos por el JSON 
process_input_video(Input_files,Output_files,Backup_files,Log_dir) #Directorios que la aplicacion "Entrada, Sadlida, Respaldo y log de mensajes para analisis"
