import cv2
import numpy as np
import pickle
from src.utils import Park_classifier


def demostration():
    """PARQUEADERO UCC
    """

    # definiendo los parametros
    rect_width, rect_height = 107, 48
    carp_park_positions_path = "data/source/CarParkPos"
    video_path = "data/source/carPark.mp4"

    # crear la instancia del clasificador que utiliza procesos de imagen básicos para clasificar
    classifier = Park_classifier(carp_park_positions_path, rect_width, rect_height)

    # Implementación de la clase
    cap = cv2.VideoCapture(video_path)
    while True:

        # leyendo el video cuadro por cuadro
        ret, frame = cap.read()

        # comprobar si hay una recuperación
        if not ret:break
        
        # procesando los marcos para preparar clasificar
        prosessed_frame = classifier.implement_process(frame)
        
        # Dibujo de aparcamientos según su estado Implementación de la clase
        denoted_image = classifier.classify(image=frame, prosessed_image = prosessed_frame)
        
        # mostrando los resultados
        cv2.imshow("Car Park Image which drawn According to  empty or occupied", denoted_image)
        
        # condición de salida
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            break
        
        if k & 0xFF == ord('s'):
            cv2.imwrite("output.jpg", denoted_image)

    # reasignación de fuentes
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demostration()
