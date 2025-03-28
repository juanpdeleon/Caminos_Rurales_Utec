import cv2
import copy
from ultralytics import solutions
import os
import shutil
import json
import re

# Diccionario de clases
CLASES = {
    1: "Moto",
    2: "Auto",
    3: "Omnibus",
    4: "C-Liviano",
    5: "C-Pesado"
}

def preocess_input_video(input_dir, output_dir, backup_dir, Log_dir):
    cropped_box = output_dir + "/cropped"
    saved_json = output_dir + "/jsonfile"

    if not os.path.exists(input_dir):
        print("El directorio original no existe..")
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(cropped_box)
            os.makedirs(saved_json)
            os.makedirs(backup_dir)
            os.makedirs(Log_dir)

        content = [files for files in os.listdir(input_dir) if files.endswith('.mp4')]

        if len(content) > 0:
            for archive in content:
                print(archive)
                path_video_file = input_dir + archive

                status = count_specific_classes(
                    path_video_file, "output_specific_classes.avi", "caminosRurales.pt",
                    [1, 2, 3,4,5], cropped_box, saved_json, archive
                )
                if status:
                    shutil.copy(path_video_file, backup_dir + "/" + archive)
                    print(f" PATH ORIGEN ########  {path_video_file} a destino {backup_dir}/{archive}")
                else:
                    print("Error procesando el archivo de video")
        else:
            print("Directorio de videos está vacío....")


def count_specific_classes(video_path, output_video_path, model_path, classes_to_count, cropped_box, saved_json, archive):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error al abrir el archivo de video"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    line_points_max = [(10, 250), (1300, 250)]  # Línea principal de detección ajustada
    counter = solutions.ObjectCounter(show=True, region=line_points_max, model=model_path, classes=classes_to_count, verbose=False, line_thickness=1)

    frame_count = 0
    objects_crossed = set()
    previous_positions = {}

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video completado o sin frames válidos.")
            break

        im0 = counter.count(im0)
       
        video_writer.write(im0)
        cv2.line(im0, (10, 250), (1300, 250), (255, 100, 255), 1)  # Línea ajustada

        frame_count += 1
        for track_id, box in zip(counter.track_ids, counter.boxes):
            x1, y1, x2, y2 = box
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

            if 250 < centroid[1] < 300:  # Rango ajustado
                if track_id in previous_positions:
                    previous_y = previous_positions[track_id][1]
                    current_y = centroid[1]

                    if current_y < previous_y:
                        direccion = "IN"
                    else:
                        direccion = "OUT"
                else:
                    direccion = "UNKNOWN"

                if track_id not in objects_crossed:
                    print(f"Objeto {track_id} cruzó la línea en dirección: {direccion}")
                    objects_crossed.add(track_id)

                    tipo_vehiculo = CLASES.get(track_id % len(CLASES), "unknown")
                   # tiempo = str(int(frame_count / fps))
                    
                    nombre_captura = save_cropped_box(im0, box, track_id, frame_count, 100, cropped_box, archive)
                    tiempo = formatear_nombre_archivo(nombre_captura)
                    list_data = [direccion, nombre_captura + ".jpg", tipo_vehiculo, 0, 'XXX0000', tiempo]

                    save_json_file(list_data, saved_json, nombre_captura)

            previous_positions[track_id] = centroid

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Reproducción detenida por el usuario.")
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    return True


def save_cropped_box(im0, box, track_id, frame_count, incremento_porcentaje, output_dir, archive):
    x1, y1, x2, y2 = map(int, box)
    ancho_original = x2 - x1
    alto_original = y2 - y1

    factor_incremento = incremento_porcentaje / 100.0
    incremento_ancho = ancho_original * factor_incremento
    incremento_alto = alto_original * factor_incremento

    centro_x = (x1 + x2) / 2
    centro_y = (y1 + y2) / 2

    nuevo_x1 = int(centro_x - (ancho_original / 2 + incremento_ancho / 2))
    nuevo_y1 = int(centro_y - (alto_original / 2 + incremento_alto / 2))
    nuevo_x2 = int(centro_x + (ancho_original / 2 + incremento_ancho / 2))
    nuevo_y2 = int(centro_y + (alto_original / 2 + incremento_alto / 2))

    cropped_box = im0[nuevo_y1:nuevo_y2, nuevo_x1:nuevo_x2]
    output_path = f"{output_dir}/{archive}_track_{track_id}_frame_{frame_count}.jpg"
    cv2.imwrite(output_path, cropped_box)
    return f"{archive}_track_{track_id}_frame_{frame_count}"


def formatear_nombre_archivo(nombre_archivo: str) -> str:
    # Buscar la fecha y hora en el nombre del archivo
    match = re.search(r'(\d{2}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d{2})', nombre_archivo)
    
    if match:
        fecha = match.group(1)
        # Se formatea la hora como hh:mm:ss.dd
        hora = f"{match.group(2)}:{match.group(3)}:{match.group(4)}.{match.group(5)}"
        return f"{fecha} {hora}"
    else:
        return "Formato inválido"
        
        
def save_json_file(list_data, saved_json, file_name):
    file_name_path = saved_json + "/" + file_name
    datos_json = {
        "direccion": list_data[0],
        "ruta": list_data[1],
        "tipo": list_data[2],
        "ejes": list_data[3],
        "matricula": list_data[4],
        "time": list_data[5]
    }
    with open(file_name_path + ".json", "w") as archivo:
        json.dump(datos_json, archivo, indent=4)

    print("Archivo JSON creado exitosamente.")
    return True

