import cv2
import copy
from ultralytics import solutions
import os
import shutil
import json


# Función principal del sistema que carga los datos
# ======================================================================================================================================
def preocess_input_video(input_dir, output_dir, backup_dir, Log_dir):  # Lee los directorios de entrada de videos y salida de los frames del algoritmo
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

        content = [files for files in os.listdir(input_dir) if files.endswith('.mp4')]  # carga la lista de videos en MP4

        if len(content) > 0:  # Si existe más de un archivo en la entrada
            for archive in content:
                print(archive)
                path_video_file = input_dir + archive

                # Procesa el archivo de video
                status = count_specific_classes(
                    path_video_file, "output_specific_classes.avi", "yolo11n.pt",
                    [1, 2, 3, 5, 7], cropped_box, saved_json, archive
                )
                if status:
                    shutil.copy(path_video_file, backup_dir + "/" + archive)
                    print(f" PATH ORIGEN ########  {path_video_file} a destino {backup_dir}/{archive}")
                else:
                    print("Error procesando el archivo de video")
        else:
            print("Directorio de videos está vacío....")


# ======================================================================================================================================

# Diccionario de clases
CLASES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


def count_specific_classes(video_path, output_video_path, model_path, classes_to_count, cropped_box, saved_json, archive):
    """Cuenta clases específicas de objetos en un video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error al abrir el archivo de video"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    line_points_max = [(10, 350), (1300, 350)]  # Línea principal de detección
    counter = solutions.ObjectCounter(show=True, region=line_points_max, model=model_path, classes=classes_to_count, verbose=False)

    frame_count = 0
    objects_crossed = set()

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video completado o sin frames válidos.")
            break

        im0 = counter.count(im0)
        video_writer.write(im0)
        cv2.line(im0, (10, 350), (1300, 350), (255, 100, 255), 1)

        frame_count += 1
        for track_id, box in zip(counter.track_ids, counter.boxes):
            x1, y1, x2, y2 = box
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

            if 350 < centroid[1] < 400:  # Detecta cruce en la región
                if track_id not in objects_crossed:
                    print(f"Objeto {track_id} cruzó la línea. Bounding Box: {box}")
                    objects_crossed.add(track_id)

                    # Determina la dirección del vehículo , por ahora weesta la linea horizontal hay que ver con la vertical
                    direccion = "I" if centroid[1] < 375 else "D"  # direccion = "IN" if centroid[1] < 375 else "OUT" se cambia IN por I y out por D
                    tipo_vehiculo = CLASES.get(track_id % len(CLASES), "unknown")

                    tiempo = str(int(frame_count / fps))
                    nombre_captura = save_cropped_box(im0, box, track_id, frame_count, 100, cropped_box, archive)
                    list_data = [direccion, nombre_captura + ".jpg", tipo_vehiculo, 0, 'XXX0000', tiempo]

                    save_json_file(list_data, saved_json, nombre_captura)

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
