<!-- PRIA - UTEC -->
<h2><span style="color: #000080;"><strong>&nbsp;Version definitiva del sistema. Marzo 2025 - Pria UTEC</strong></span></h2>
<p><b></b>Se incluyen todas las actualizaciones del codigo al momento de la entrega del informe final.- Ultima actualización : 31-03-2025  Carpeta "Marzo 2025"</b></p>
<ul>
<li>
<h3><strong>Objetivo principal</strong></h3>
<ul>
<li>Analiza todos los videos y para cada deteccion de vehiculo, genera un archivo JSON con la informacion relevante para el DASHBOARD.</li>
<li>Formato estipulado de JSON
<ul>
<li>Direccion: IN o OUT o una derivacion de las mismas</li>
<li>Nombre del archivo JPG con la deteccion, el mismo es dado por el nombre del video origen, mas el frame en el que fue detectado el vehiculo, ya que un video puede tener mas de 1 deteccion.</li>
<li>Tipo de Vehiculo : <br />
<ul>
<li style="list-style-type: none;">
<ul>
<li style="list-style-type: none;">
<ul>
<li>MOTO</li>
<li>AUTO</li>
<li>OMNIBUS</li>
<li>CAMIONL (Camion Liviano)</li>
<li>CAMIONP ( Camion Pesado):</li>
</ul>
</li>
</ul>
</li>
</ul>
</li>
<li>Ejes del vehiculo</li>
<li>Matricula, de momento no reconocido</li>
<li>Time, tomado del la camara.</li>
</ul>
</li>
</ul>
</li>
<li>
<h3><strong>Estructura del algoritmo</strong></h3>
<ul>
<li>Archivo de configuraciones "<span style="text-decoration: underline;"><strong>config.json</strong></span>"
<ul>
<li>&nbsp;Formato : <br />
<ul>
<li>{<br />"Input_files": "entrada/",&nbsp;&nbsp; -&gt; Ubicacion de los videos de entrada<br />"Output_files": "./salidas", -&gt; ubicacion de las todas las salidas del sistema<br />"Backup_files": "./backup", -&gt; Ubicacion a donde se mueven los videos de entrada una vez procesados<br />"Log_dir": "./log_dir", } -&gt; Log del Sistema</li>
</ul>
</li>
</ul>
</li>
<li>Archivo de funciones : "<strong><span style="text-decoration: underline;">libreria.py</span></strong>" es el que tiene toda la logica del YOLO y las detecciones.</li>
<li>Archivo de Inicio : "<span style="text-decoration: underline;"><strong>inicio.py</strong></span>" El el archivo que se ejecuta desde python, el&nbsp; es responsable de leer el archivo de configuracion del sistema e informar a las funciones de que manera se va a procesar.</li>
</ul>
</li>
</ul>




# **Trabajo UTEC** - DICIEMBRE 20224 <br>
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
                  
            
