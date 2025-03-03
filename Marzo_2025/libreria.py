import cv2
import os
import shutil
import json
import re
from ultralytics import solutions

# Diccionario de clases
CLASES = {
   0: "Auto",
   1: "CamionL",
   2: "CamionP",
   3: "Moto",
   4: "Omnibus"
}

def process_input_video(input_dir, output_dir, backup_dir, Log_dir):
    """Procesa los videos en el directorio de entrada y genera salidas."""
    cropped_box = os.path.join(output_dir, "cropped")
    saved_json = os.path.join(output_dir, "jsonfile")
    saved_csv = os.path.join(output_dir, "jsonfile")

    if not os.path.exists(input_dir):
        print("El directorio original no existe.")
        return

    # Crear directorios si no existen
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cropped_box, exist_ok=True)   
    os.makedirs(saved_json, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(Log_dir, exist_ok=True)

    # Obtener archivos de video
    content = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

    if not content:
        print("Directorio de videos está vacío.")
        return

    for archive in content:
        print(f"Procesando: {archive}")
        path_video_file = os.path.join(input_dir, archive)

        status = count_specific_classes(
            path_video_file, "output_specific_classes.avi", "best.pt",
            [0,1,2,3,4], cropped_box, saved_json,saved_csv, archive
        )

        if status:
            shutil.copy(path_video_file, os.path.join(backup_dir, archive))
            print(f"Video copiado: {path_video_file} -> {backup_dir}/{archive}")
        else:
            print("Error procesando el archivo de video.")
            
            
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
        show=True, region=line_points_max, model=model_path,
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
                            list_data = [direccion, nombre_captura + ".jpg", tipo_vehiculo, 0, 'XXX0000', tiempo, class_id]
                            
                            # Llamar a la nueva función que guarda tanto el JSON como el CSV
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


import csv
import json
import os

def save_json_file_and_csv(list_data, saved_json, saved_csv, file_name, video_path,box, precision,archive_class):
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
        class_name = list_data[2]  # Nom"5_ND_"+archivebre de la clase detectada
        class_id = list_data[6]    # ID de la clase
        direction = list_data[0]   # Dirección (IN/OUT/UNKNOWN)
        bbox_coords = ', '.join(map(str, box))  # Coordenadas del bounding box
        detection_precision = precision  # Precisión de la detección

        # Escribir una nueva línea con los datos de la detección
        writer.writerow([video_file, image_file, class_name, class_id, direction, bbox_coords, detection_precision])

    print(f"Datos guardados en el archivo CSV: {csv_file_path}")

    return True