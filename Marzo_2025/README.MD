<!-- ####### HEY, I AM THE SOURCE EDITOR! #########-->
<h2 style="color: #2e6c80;">&nbsp;Version definitiva del sistema. Marzo 2025 - Pria UTEC</h2>
<p>Se incluyen todas las actualizaciones del codigo al momento de la entrega del informe final.</p>
<ul>
<li>Objetivo principal
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
<li>Estructura del algoritmo
<ul>
<li>Archivo de configuraciones "config.json"
<ul>
<li>&nbsp;Formato : <br />
<ul>
<li>{<br />"Input_files": "entrada/",&nbsp;&nbsp; -&gt; Ubicacion de los videos de entrada<br />"Output_files": "./salidas", -&gt; ubicacion de las todas las salidas del sistema<br />"Backup_files": "./backup", -&gt; Ubicacion a donde se mueven los videos de entrada una vez procesados<br />"Log_dir": "./log_dir", } -&gt; Log del Sistema</li>
</ul>
</li>
</ul>
</li>
<li>Archivo de funciones : "libreria.py" es el que tiene toda la logica del YOLO y las detecciones.</li>
<li>Archivo de Inicio : "inicio.py" es el&nbsp; responsable de leer el archivo de configuracion del sistema e informar a las funciones de que manera se va a procesar.</li>
</ul>
</li>
</ul>
