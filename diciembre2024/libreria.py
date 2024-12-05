import cv2
import copy
from ultralytics import solutions
import os
import shutil
import json


# Funcion principal del sistema que carga los datos
#======================================================================================================================================
def preocess_input_video(input_dir,output_dir,backup_dir,Log_dir): # Lee los directorios de entrada de videos  y salida del los frames del algoritmo
    #estos directoriose se crean dentro de la carpeta output
    cropped_box=output_dir+"/cropped"
    saved_json=output_dir+"/jsonfile"

    if not os.path.exists(input_dir):
      print("El directorio original no existe..")
    else:
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)     #Si no existe el output_dir crea la extructura de todos los box y los frames
            os.makedirs(cropped_box)    #Directorio del boundingbox recortado en formato JPG
            os.makedirs(saved_json)    #Directorio del json completo de la detecction
            os.makedirs(backup_dir)     # directorio a donde se mueve el video procesado
            os.makedirs(Log_dir)        # Directorio a dode se guardara un archivo con toda los informes del algoritmo
        
        content =  [files for files in os.listdir(input_dir) if files.endswith('.mp4')] # carga la lista de videos en MP4
       
        if len(content)>=0:  # Si existe mas de un archivo en la entrada
        #Carga el sistema principal
            for archive in content:
                print(archive)
                path_video_file = input_dir+archive

                #//classes Indentificadas por yolo ,
                 #   "1": "bicycle",
                 #  "2": "car",
                 # "3": "motorcycle",
                 #"5": "bus",
                 #"7": "truck",

                status=count_specific_classes(path_video_file, "output_specific_classes.avi", "yolo11n.pt", [1,2,3,5,7], cropped_box, saved_json,archive)              
                status = True
                if status:
                    # os.remove(archive)
                    shutil.copy(path_video_file, backup_dir+"/"+archive) #hace un respaldo de video procesado
                    print(f" PATH ORIGEN ########  {path_video_file} a destino {backup_dir}/{archive}")
                else:
                    print("Error procesando el archivs de video")
        else:
            print("Directorio de videos esta vacio....")

#======================================================================================================================================

def save_cropped_box(im0, box, track_id, frame_count,incremento_porcentaje, output_dir,archive):
    # Guarda los bounding boxes de captura
    # Cconvierte los valores en enteros     
    x1, y1, x2, y2 = map(int, box)  # Ensure the coordinates are integers
    ancho_original = x2 - x1
    alto_original = y2 - y1
    
    # Convierte el porcentaje de incremento a un factor (por ejemplo, 50% a 0.5)
    factor_incremento = incremento_porcentaje / 100.0
    
    # Calcula el aumento en dimensiones
    incremento_ancho = ancho_original * factor_incremento
    incremento_alto = alto_original * factor_incremento
    
    # Encuentra el centro del bounding box
    centro_x = (x1 + x2) / 2
    centro_y = (y1 + y2) / 2
    
    # Calcula las nuevas coordenadas
    nuevo_x1 = int(centro_x - (ancho_original / 2 + incremento_ancho / 2))
    nuevo_y1 = int(centro_y - (alto_original / 2 + incremento_alto / 2))
    nuevo_x2 = int(centro_x + (ancho_original / 2 + incremento_ancho / 2))
    nuevo_y2 = int(centro_y + (alto_original / 2 + incremento_alto / 2))
 
    cropped_box = im0[nuevo_y1:nuevo_y2, nuevo_x1:nuevo_x2]  # Corta la imagen con una apliacion
    output_path = f"{output_dir}/{archive}_track_{track_id}_frame_{frame_count}.jpg"
    cv2.imwrite(output_path, cropped_box)  #
    return f"{archive}_track_{track_id}_frame_{frame_count}"



#Cambia la estructura del diccionario de Yolo para un formato que puedo procesar
#======================================================================================================================================
def make_hashable(dict_items):
    return {(k, tuple(v.items()) if isinstance(v, dict) else v) for k, v in dict_items}

#Guarda el frame del video de la deteccion, el primer frame que se detecta el objeto sobre la linea de track de yolo (De momento se descontinuaria) 
#======================================================================================================================================
def save_detection_frame(frame, output_dir, frame_count,archive):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(f"{output_dir}/{archive}_frame_{frame_count}.jpg", frame)

#Fucnion Principal que procesa los video y hace las llamadas
#======================================================================================================================================
def count_specific_classes(video_path, output_video_path, model_path, classes_to_count, cropped_box, saved_json,archive):
    """Count specific classes of objects in a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
  
    in_value_old=0
    out_value_old=0
    
    line_points_max= [(10,350),(1300,350)] #Lineas de deteccion real de yolo
    line_points_min=[(10,400),(1300,400)] # linea de deteccion secuntaria, para manejar el espectro de errores de YOLO.
    
    counter = solutions.ObjectCounter(show=True, region=line_points_max, model=model_path, classes=classes_to_count,verbose=False)
    frame_count = 0

    previous_items = make_hashable(counter.classwise_counts.items())
    previous_positions = {}  # Gurda la prosicion previa
    objects_crossed = set()  # Guarda los vehiculos que pasaron la linea
    # file_data = open("test.csv", "a")
    while cap.isOpened():
        success, im0 = cap.read()
            
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        
        im0 = counter.count(im0)        
        video_writer.write(im0)
        cv2.line(im0,(10,350),(1300,350),(255,100,255),1)

        frame_count+=1
        current_items = make_hashable(counter.classwise_counts.items())
        added = current_items - previous_items # obtengo el valore agregado por diferencia
        temp_vehicle = []
        list_data=[]
        tiempo=0
        if added:
            print(f" Vehiculos agregados : {added}")
             #archio de salia del video
            
            #save_detection_frame(im0, saved_frame, frame_count,archive)
            previous_items = current_items   
            direccion="X"
            for track_id, box in zip(counter.track_ids, counter.boxes):
                x1, y1, x2, y2 = box               
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                # Calcula el ancho y alto original                              
                centro = centroid[1]
                
                if (centro > 350 and centro <400):  # Direccion de deteccion arrriba y abajo 
                        if track_id not in objects_crossed:
                            print(f"Object {track_id} crossed the box! Bounding Box: {box}")
                            name_archive_crop=save_cropped_box(im0, box, track_id, frame_count,100,cropped_box,archive)
                            objects_crossed.add(track_id)
                            tiempo = str(int(frame_count/30)) #falta tener el nombre y ahi se calcula la cantidad de frames sobre 30
                            output_csv = f"{cropped_box}/{archive}_csv_{track_id}_frame_{frame_count}.csv"
                            elements = list(added)[0]
    
                            # Extraer los valores directamente
                            type_vehicle = elements[0]  
                      
                            if type_vehicle not in temp_vehicle:
                                temp_vehicle = [type_vehicle, elements[1][0][1], elements[1][1][1]]
                                print(f"SALLIDA DE TEMP : {temp_vehicle}")
                           
                            
                            in_value = (elements[1][0][1]) #-(in_value_old) # 0
                            out_value = (elements[1][1][1])#-(out_value_old) # 1
                            print(f"SALLIDA DE OBJECT: {objects_crossed}")
                            
                            if in_value==1
                                direccion="D"
                            if out_value==1:
                                direccion="I"
                            #genera una linea en cada archivo para cada detecccioon, puede ser que 1 video tenga mas que 1 vehiculo
                          #  file_data = open(output_csv, "a")
                            
                            #file_data.write(name_archive_crop+','+str(type_vehicle)+','+str(in_value)+','+str( out_value)+'\n') 
                            list_data= [direccion,name_archive_crop+".jpg",str(type_vehicle),0,'XXX0000',tiempo]
                            print(list_data)
                            
                           # file_data.close()
                            save_json_file(list_data,saved_json,name_archive_crop)
                            list_data.clear()# vacia la lista

                            in_value_old=elements[1][0][1]
                            out_value_old=elements[1][1][1]
                            
                previous_positions[track_id] = centroid
                
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Reproducción detenida por el usuario.")
            break
            
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()



    # PARTE DEL SISTEMA QUE SE VA A BORRAR; SE UTILIZA PARA TRABAJAR EN CONSOLA 
    #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Parametros de salida obcionales // para control durante el proceso de configuracion del algoritmo
    print("Resumen --------------------------")
     # Imprime los totales
    # Inicializa un diccionario para los totales
    total_counts = {}
   #archio de salia del video
   
    # Itera sobre cada tipo de vehículo en el diccionario
    for vehicle, data in counter.classwise_counts.items():
        # Suma las entradas y salidas
        total = data['IN'] + data['OUT']

    #file_data.close()
    
    # Guarda el total en el nuevo diccionario
    total_counts[vehicle] = total
    print("Listado Final :")
    # Imprime los totales por tipo en la consola para control
    for vehicle, total in total_counts.items():
        print(f" Listado TOTAL DE VEHICULOS : {vehicle.capitalize()}s: {total}")

    #se debe implementar una funcion de control para que si no da errores el algoritmo retornara TRUE
    # si da falla retornara FALSE 
    return True
        
def save_json_file(list_data,saved_json,file_name): # recibo como parametros los datos en una lista y el nombre del archivo 
    # Lista de datos, cada sublista representa un conjunto de datos
    file_name_path=saved_json+"/"+file_name
    datos_json = {
            "direccion": list_data[0],
            "ruta": list_data[1],
            "tipo": list_data[2],
            "ejes": list_data[3],
            "matricula": list_data[4],
            "time": list_data[5]
        }
    with open(file_name_path+".json", "w") as archivo:
        json.dump(datos_json, archivo, indent=4)
        
    print("Archivo JSON creado exitosamente.")
    print(file_name_path)

    return True 

def read_Json_timestamp(file_name):
    
    return true
    