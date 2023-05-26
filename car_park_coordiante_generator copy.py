import cv2
from src.utils import Coordinate_denoter

def demostration():
   """Es la demostración del car_park_coordinate_generatorçpy .
    """

    
    # creando la instancia de Coordinate_generator para extraer las coordenadas del estacionamiento
    coordinate_generator=Coordinate_denoter()

    # lectura e inicialización de las coordenadas
    coordinate_generator.read_positions()

    # establecer las variables iniciales
    image_path = "data/source/example_image.png"
    rect_width, rect_height = coordinate_generator.rect_width, coordinate_generator.rect_height

    # sirviendo la ventana GUI hasta que el usuario la finalice
    while True:
        
        # refrescando la imagen
        image =cv2.imread(image_path)

        # dibujar las coordenadas actuales del aparcamiento
        for pos in coordinate_generator.car_park_positions: 
            
            # definiendo los límites
            start = pos
            end = (pos[0]+rect_width, pos[1]+rect_height)

            #dibujar el rectángulo en la imagen
            cv2.rectangle(image,start,end,(0,0,255),2)
        
        cv2.imshow("Image",image)

        # vincular la devolución de llamada del mouse
        cv2.setMouseCallback("Image",coordinate_generator.mouseClick)

        # condición de salida
        if cv2.waitKey(1) == ord("q"):
            break

    # reasignar las fuentes
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demostration()