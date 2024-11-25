# **Trabajo UTEC**<br>
## **Camineria Rural , Tracking de vehiculos on IA** <br>
### ** El trabajo esta echo en Conda y Jupiter **
El archivo principal **"camineria_25_11.ipynb"** de trabajo esta escrito en Jupiter con la finalidad de hacer pruebas.-<br>
Las funciones estan en un archivo llamado **libreria.py**
los parametros de entrada estan en el archivo **config.json** con el siguiente formato.
{
    "Input_files": "entrada/",
    "Output_files": "./salidas",
    "Backup_files": "./backup",
	"Log_dir":"./log_dir"
}

Para cada detecccion se crean 2 archios:<br>
1 ".jpg" con el crop de la deteccion con el formato : video_toten1_14-11-2024_01_11.mp4_track_1_frame_4.jpg de donde : video_toten1_14-11-2024_01_11.mp4 seria el nombre original del video yy despues informacion del track<br>
1 json con la informacion de la deteccion en el formato :

{
    "direccion": "D",
    "ruta": "video_toten1_14-11-2024_01_11.mp4_track_38_frame_106.jpg",
    "tipo": "car",
    "ejes": 0,
    "matricula": "XXX0000",
    "time": "3"
}

[ ]Mejorar el algoritmo de trabajo y guardado del json.<br>
[ ]Programar la salida de un archivo log, con los mensajes del algorimo <br>
[ ]Probar con modelos entrenados especificoss <br>
                  
            
