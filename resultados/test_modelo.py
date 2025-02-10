import os
import json
import cv2

def cargar_datos(imagenes_dir, json_dir, salida_dir):
    if not os.path.exists(salida_dir):
        os.makedirs(salida_dir)
    
    for archivo in os.listdir(imagenes_dir):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta_imagen = os.path.join(imagenes_dir, archivo)
            nombre_json = os.path.splitext(archivo)[0] + ".json"
            ruta_json = os.path.join(json_dir, nombre_json)
            
            if os.path.exists(ruta_json):
                with open(ruta_json, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                guardar_imagen_modificada(ruta_imagen, datos, salida_dir, archivo)
            else:
                print(f"No se encontró el JSON para la imagen: {archivo}")

def guardar_imagen_modificada(ruta_imagen, datos, salida_dir, nombre_original):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"No se pudo cargar la imagen: {ruta_imagen}")
        return
    
    tipo = datos.get("tipo", "Desconocido")
    cv2.putText(imagen, tipo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    ruta_salida = os.path.join(salida_dir, nombre_original)
    cv2.imwrite(ruta_salida, imagen)
    print(f"Imagen guardada en: {ruta_salida}")

directorio_imagenes = "C:/Users/jpdeleon/Utec/Camineria_Rural/salidas/cropped"  # Reemplazar con la ruta real  C:\Users\jpdeleon\Utec\Camineria_Rural\salidas\jsonfile
directorio_json = "C:/Users/jpdeleon/Utec/Camineria_Rural/salidas/jsonfile"
directorio_salida = "C:/Users/jpdeleon/Utec/Camineria_Rural/salidas/catalogado"  # Reemplazar con la ruta real
cargar_datos(directorio_imagenes, directorio_json, directorio_salida)#ruta/a/json