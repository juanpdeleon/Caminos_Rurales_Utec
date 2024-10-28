def convertir(input_file, output_file, width=1024, height=768):
    # Captura el video de entrada
    cap = cv2.VideoCapture(input_file)

    # Verifica si se abrió el video correctamente
    if not cap.isOpened():
        print(f"No se pudo abrir el archivo: {input_file}")
        return

    # Obtener la resolución original
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Resolución original: {original_width}x{original_height}")

    # Si la resolución es mayor que 1024x768, redimensionar
    if original_width > width or original_height > height:
        print("Redimensionando video...")
        # Obtener la tasa de cuadros y el codec del video original
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para archivos MP4

        # Crear el objeto VideoWriter para el video de salida
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Cambiar la resolución del frame
            resized_frame = cv2.resize(frame, (width, height))

            # Escribir el frame redimensionado en el video de salida
            out.write(resized_frame)

        # Liberar los objetos
        out.release()
        print(f"Video redimensionado guardado como: {output_file}")
    else:
        print("La resolución es adecuada. No se necesita redimensionar.")

    # Liberar el objeto de captura
    cap.release()