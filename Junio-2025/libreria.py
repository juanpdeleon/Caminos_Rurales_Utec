#***************************************************
#
# Proyecto: [Camineria Rural]
# Archivo: [libreria.py]
# Descripción: [Proyecto de camineria rural para la deteccion de vehiculos de distintos portes]
#
# Autores:
# - [Bruna de Vargas] ([bruna.devargas@utec.edu.uy])
# - [Pablo Cuña] ([pablo.cuna@utec.edu.uy])
# - [Juan Pedro de León] ([juan.deleon@utec.edu.uy])
# - [Victor Castelli] ([victor.castelli@utec.edu.uy])
# 
# Institución: [UTEC - ITRN]
# Departamento: [PRIA - Postrado en Robotia e Intelifencia Artificial]
# Fecha: [13-03-2025]
#
# **************************************************
import cv2
import os
import shutil
import json
import re
from ultralytics import solutions
from ultralytics import YOLO
import time  # Importar para esperar 30 segundos
import csv  # Aquí agregamos la importación del módulo csv
import torch
import traceback


#Modelo de ejes
#axis_model = YOLO("vehiculos.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("vehiculos.pt").to(device)  # Enviar modelo a GPU
axis_model = YOLO("axis.pt").to(device)

# Diccionario de clases
CLASES = {
   0: "Auto",
   1: "CamionL",
   2: "CamionP",
   3: "Moto",
   4: "Omnibus"
}



def process_input_video(input_dir, output_dir, backup_dir, log_dir):
    """Procesa los videos en el directorio de entrada y genera salidas. Incluye manejo de errores robusto."""
    cropped_box = os.path.join(output_dir, "cropped")
    saved_json = os.path.join(output_dir, "jsonfile")
    saved_csv = os.path.join(output_dir, "jsonfile")

    for folder in [output_dir, cropped_box, saved_json, saved_csv, backup_dir, log_dir]:
        os.makedirs(folder, exist_ok=True)

    print("Monitoreando directorio de entrada...")

    while True:
        try:
            videos = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
            if not videos:
                print("Directorio de videos vacío.")
            else:
                for video in videos:
                    try:
                        print(f"\nProcesando video: {video}")
                        path_video_file = os.path.join(input_dir, video)

                        status = count_specific_classes(
                            path_video_file, "output_specific_classes.avi", "vehiculos.pt",
                            [0, 1, 2, 3, 4], cropped_box, saved_json, saved_csv, video
                        )

                        if status:
                            shutil.copy(path_video_file, os.path.join(backup_dir, video))
                            os.remove(path_video_file)
                            print(f" Video procesado exitosamente: {video}")
                        else:
                            print(f"️ El procesamiento de {video} falló.")

                    except Exception as ve:
                        log_path = os.path.join(log_dir, f"error_{video}.log")
                        with open(log_path, "w") as log_file:
                            log_file.write(traceback.format_exc())
                        print(f" Error procesando {video}. Detalles en {log_path}")

            print(" Esperando 30 segundos para nuevos archivos...")
            time.sleep(30)

        except KeyboardInterrupt:
            print("Proceso interrumpido por el usuario.")
            break
        except Exception as e:
            print(f"Error inesperado: {e}")
            time.sleep(30)


def count_specific_classes(video_path, output_video_path, model_path, classes_to_count, cropped_box, saved_json, saved_csv, archive):
    """Cuenta clases específicas de objetos en un video y guarda los resultados en JSON y CSV."""
    clasificadas = "./salidas/clasificadas"
    
    cap = cv2.VideoCapture(video_path)
    
    # Capturar el ancho y alto del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_size = (width * height) * 0.80  # 80% del tamaño del video será el máximo bounding box válido

    assert cap.isOpened(), "Error al abrir el archivo de video"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Nueva línea de detección vertical en el centro del video
    line_x = width // 2
    line_points_max = [(line_x, 0), (line_x, height)]  # Línea vertical desde arriba hasta abajo

    counter = solutions.ObjectCounter(
        show=False, region=line_points_max, model=model_path,
        classes=classes_to_count, verbose=False, line_thickness=0, show_in=False, show_out=False, line_width=0
    )

    frame_count = 0
    objects_crossed = set()
    previous_positions = {}

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video completado o sin frames válidos.")
            break

        imagen_nueva = im0.copy()

        # Procesar el frame con YOLO
        results = counter.model(im0)
        im0 = counter.count(im0)

        video_writer.write(im0)
        frame_count += 1

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []
            confidences = result.boxes.conf.cpu().numpy()  # Confianza de cada detección

            for track_id, box, class_id, confidence in zip(track_ids, boxes, class_ids, confidences):
                x1, y1, x2, y2 = box
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                tipo_vehiculo = CLASES.get(class_id, "unknown")

                if (line_x - 5) < centroid[0] < (line_x + 5):  # Detección en la línea vertical
                    if track_id in previous_positions:
                        previous_x = previous_positions[track_id][0]
                        current_x = centroid[0]

                        if current_x < previous_x:
                            direccion = "IN"
                        else:
                            direccion = "OUT"
                    else:
                        direccion = "UNKNOWN"

                    if track_id not in objects_crossed:
                        print(f"Objeto {track_id} cruzó la línea en dirección: {direccion}")
                        objects_crossed.add(track_id)

                        box_size = (((x1-x2)*(y1-y2)))  # Tamaño del bounding box
                        if box_size < video_size:  # Si el box es menor al 80% del video, guardar imagen
                            nombre_captura = save_cropped_box(im0, box, track_id, frame_count, 100, cropped_box, archive)
                            #archive_class=archive
                            if class_id==0:                           
                                archive_class='0_Auto_'+archive                                
                                nombre_captura_limpia = save_cropped_box(imagen_nueva, box, track_id, frame_count, 0, clasificadas,archive_class) # captura de pantalla limpia sin bo box
                            if class_id==1:                    
                                archive_class="1_CamionL_"+archive                               
                                nombre_captura_limpia = save_cropped_box(imagen_nueva, box, track_id, frame_count, 0, clasificadas,archive_class) # captura de pantalla limpia sin bo box
                            if class_id==2:
                                archive_class="2_CamionP_"+archive
                                nombre_captura_limpia = save_cropped_box(imagen_nueva, box, track_id, frame_count, 0, clasificadas,archive_class) # captura de pantalla limpia sin bo box
                            if class_id==3:
                                archive_class="3_Moto_"+archive
                                nombre_captura_limpia = save_cropped_box(imagen_nueva, box, track_id, frame_count, 0, clasificadas,archive_class) # captura de pantalla limpia sin bo box
                            if class_id==4:
                                archive_class="4_Omnibus_"+archive
                                nombre_captura_limpia = save_cropped_box(imagen_nueva, box, track_id, frame_count, 0, clasificadas,archive_class) # captura de pantalla limpia sin bo box
                            if class_id>4:
                                archive_class="5_ND_"+archive
                                nombre_captura_limpia = save_cropped_box(imagen_nueva, box, track_id, frame_count, 0, clasificadas,archive_class) # captura de pantalla limpia sin bo box
                                
                            #nombre_captura = save_cropped_box(im0, box, track_id, frame_count, 100, cropped_box, archive)
                            tiempo = formatear_nombre_archivo(nombre_captura)
                                                       
                            # Llamar a la nueva función que guarda tanto el JSON como el CSV                          
                            #
                            # llamar a la funciona que devuelve la cantidad de neumaticos
                          
                            if class_id==1 or class_id==2 or class_id==3:
                                print(" --------------------------------------------************",nombre_captura_limpia)
                                axis = search_axis(clasificadas,nombre_captura_limpia,axis_model)                                
                            else:
                                axis=2

                            list_data = [direccion, nombre_captura + ".jpg", tipo_vehiculo,axis, 'XXX0000', tiempo, class_id]
                            save_json_file_and_csv(list_data, saved_json, saved_csv, nombre_captura,video_path, box, confidence,archive_class)

                previous_positions[track_id] = centroid

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Reproducción detenida por el usuario.")
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    return True

def save_cropped_box(im0, box, track_id, frame_count, incremento_porcentaje, output_dir, archive):
    """Guarda una imagen recortada del bounding box."""
    output_path = os.path.join(output_dir, f"{archive}_track_{track_id}_frame_{frame_count}.jpg")
    if incremento_porcentaje>0:
        x1, y1, x2, y2 = map(int, box)
        ancho_original = x2 - x1
        alto_original = y2 - y1

        factor_incremento = incremento_porcentaje / 100.0
        incremento_ancho = ancho_original * factor_incremento
        incremento_alto = alto_original * factor_incremento

        centro_x = (x1 + x2) / 2
        centro_y = (y1 + y2) / 2

        nuevo_x1 = max(0, int(centro_x - (ancho_original / 2 + incremento_ancho / 2)))
        nuevo_y1 = max(0, int(centro_y - (alto_original / 2 + incremento_alto / 2)))
        nuevo_x2 = min(im0.shape[1], int(centro_x + (ancho_original / 2 + incremento_ancho / 2)))
        nuevo_y2 = min(im0.shape[0], int(centro_y + (alto_original / 2 + incremento_alto / 2)))

        cropped_box = im0[nuevo_y1:nuevo_y2, nuevo_x1:nuevo_x2]        
        cv2.imwrite(output_path, cropped_box)
    else:
        cv2.imwrite(output_path, im0)
    return f"{archive}_track_{track_id}_frame_{frame_count}"

def formatear_nombre_archivo(nombre_archivo: str) -> str:
    """Formatea el nombre del archivo para extraer fecha y hora."""
    match = re.search(r'(\d{2}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d{2})', nombre_archivo)
    if match:
        fecha = match.group(1)
        hora = f"{match.group(2)}:{match.group(3)}:{match.group(4)}.{match.group(5)}"
        return f"{fecha} {hora}"
    else:
        return "Formato inválido"

def save_json_file_and_csv(list_data, saved_json, saved_csv, file_name, video_path, box, precision, archive_class):
    """Guarda los datos en un archivo JSON y también en un archivo CSV."""
    # Primero, guardar los datos en el archivo JSON como ya se hace
    json_file_path = os.path.join(saved_json, f"{file_name}.json")
    datos_json = {
        "direccion": list_data[0],
        "ruta": list_data[1],
        "tipo": list_data[2],
        "ejes": list_data[3],
        "matricula": list_data[4],
        "time": list_data[5]
    }
    with open(json_file_path, "w") as archivo:
        json.dump(datos_json, archivo, indent=4)

    print(f"Archivo JSON creado: {json_file_path}")
    
    # Ahora, guardar los datos también en el archivo CSV
    csv_file_path = os.path.join(saved_csv, "detections.csv")
    
    # Si el archivo CSV no existe, crear la cabecera
    header = ["video_file", "image_file", "class_name", "class_id", "direction", "bbox_coords", "detection_precision"]
    
    # Abrir el archivo CSV en modo append (agregar líneas)
    with open(csv_file_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Si el archivo está vacío, escribir la cabecera
        if csvfile.tell() == 0:
            writer.writerow(header)
        
        # Extraer la información de la lista de datos
        video_file = os.path.basename(video_path)
        image_file = archive_class  # Ruta de la imagen
        class_name = list_data[2]  # Nombre de la clase detectada
        class_id = list_data[6]    # ID de la clase
        direction = list_data[0]   # Dirección (IN/OUT/UNKNOWN)
        bbox_coords = ', '.join(map(str, box))  # Coordenadas del bounding box
        detection_precision = precision  # Precisión de la detección

        # Escribir una nueva línea con los datos de la detección
        writer.writerow([video_file, image_file, class_name, class_id, direction, bbox_coords, detection_precision])

    print(f"Datos guardados en el archivo CSV: {csv_file_path}")

    return True

def search_axis(clasificadas, archive_class,axis_model):
    
    #output_path = os.path.join(output_dir, f"{archive}_track_{track_id}_frame_{frame_count}.jpg")
    image_path = os.path.join(clasificadas, f"./{archive_class}.jpg")
    image_path = os.path.normpath(image_path)
    image_path = image_path.replace("\\", "/")
   
    print("*********************************************************************************************************************")
    print("Procesando imagen:", image_path)
    print(archive_class)
    print("****************************")

    # Verificar si la imagen existe antes de intentar leerla
    if not os.path.exists(image_path):
        print("Error: La imagen no existe.")
        return 0

    image = cv2.imread(image_path)
    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return 0

    # Convertir a RGB para YOLO
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Realizar la detección
    results = axis_model.predict(image, conf=0.37)
    #resultados = modelo.predict(imagen, conf=0.37, device=dispositivo)

    # Verificar si hay detecciones
    if not results or len(results) == 0:
        print("No se detectaron objetos.")
        return 2

    # Obtener bounding boxes detectadas
    detections = results[0].boxes if results[0].boxes is not None else []

    # Contar el número de objetos detectados
    num_objects = max(len(detections), 2)
    if num_objects<2:
        num_objects=2
        
    return num_objects

    