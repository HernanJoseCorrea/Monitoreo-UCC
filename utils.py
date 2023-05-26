import cv2
import pickle
import numpy as np

class Park_classifier():
    """Simplemente utiliza métodos de procesamiento de imágenes digitales en lugar de aprendizaje profundo para clasificar si el espacio de estacionamiento está vacío o no.
    """
    
    def __init__(self, carp_park_positions_path:pickle, rect_width:int=None, rect_height:int=None):
        self.car_park_positions = self._read_positions(carp_park_positions_path) 
        self.rect_height = 48 if rect_height is None else rect_height
        self.rect_width = 107 if rect_width is None else rect_width
    
    
    def _read_positions(self, car_park_positions_path:pickle)->list:
        """Lee el archivo pickle para evitar cualquier corrupción o error de datos.

        Devoluciones
        -------
        lista
            Lista de las tuplas que almacena las coordenadas del punto superior izquierdo del rectángulo del aparcamiento. Ejemplo de demostración: [(x_1, y_1), ..., [x_n, y_n]]
        """

        car_park_positions = None
        try:
            car_park_positions = pickle.load(open(car_park_positions_path, 'rb'))
        except Exception as e:
            print(f"Error: {e}\n It raised while reading the car park positions file.")

        return car_park_positions

    def classify(self, image:np.ndarray, prosessed_image:np.ndarray,threshold:int=900)->np.ndarray:
        """Recorta la imagen ya procesada en regiones de estacionamiento y clasifica el espacio de estacionamiento como vacío o no según el umbral.

        Parámetros
        ----------
        imagen: np.ndarray
            Imagen que ya está procesada por los métodos de procesamiento de imágenes digitales opencv para preparar la clasificación.
        umbral: int, opcional
            Es el valor límite para clasificar las imágenes ya procesadas, por defecto 900

        Devoluciones
        -------
        np.ndarray
            Imagen que ha dibujado según su clase.
        """
        # Averiguar las plazas de aparcamiento vacías y ocupadas y dibujarlas.
        empty_car_park = 0
        for x, y in self.car_park_positions:
            
           # definir los puntos inicial y final del rectángulo como una línea cruzada
            col_start, col_stop = x, x + self.rect_width
            row_start, row_stop = y, y + self.rect_height

            # recortar la imagen del formulario de las áreas de estacionamiento
            crop=prosessed_image[row_start:row_stop, col_start:x+col_stop]
            
            # contando el número de píxeles que están por debajo del valor umbral debido a la expectativa de que los pasos de procesamiento de imágenes anteriores
            count=cv2.countNonZero(crop)
            
            # clasificación según el valor umbral para actualizar los recuentos y configurar los parámetros de dibujo
            empty_car_park, color, thick = [empty_car_park + 1, (0,255,0), 5] if count<threshold else [empty_car_park, (0,0,255), 2]
                
            # dibujar el rectángulo en la imagen
            start_point, stop_point = (x,y), (x+self.rect_width, y+self.rect_height)
            cv2.rectangle(image, start_point, stop_point, color, thick)
        
        
        # dibujar el rectángulo de la leyenda en el lado izquierdo de la imagen
        cv2.rectangle(image,(45,30),(500,78),(300,0,180),-1)

        ratio_text = f'-*UCC*-Espacios libres:: {empty_car_park}/{len(self.car_park_positions)}'
        cv2.putText(image,ratio_text,(50,60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        
        return image
        
    def implement_process(self, image:np.ndarray)->np.ndarray:
        """Procesa la imagen aplicando métodos de procesamiento de imagen digital opencv.

        Parámetros
        ----------
        imagen: np.ndarray
            Imagen de destino que se procesará para clasificar previamente.

        Devoluciones
        -------
        np.ndarray
            Imagen procesada.
        """
        # definir el tamaño del parámetro de matriz del kernel
        kernel_size=np.ones((3,3),np.uint8)

       # escala de grises para reducir el canal de color.
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

       # desenfoque gaussiano para reducir el ruido
        blur=cv2.GaussianBlur(gray, (3,3), 1)
        
        # implementing threashold to get forground object
        Thresholded=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
        
       # desenfoque de la imagen para reducir el ruido y normalizar la brecha de valor de píxel causada por el umbral adaptativo
        blur=cv2.medianBlur(Thresholded, 5)

        #dilatación para aumentar el objeto de primer plano.
        dilate=cv2.dilate(blur,kernel_size,iterations=1)

        return dilate

class Coordinate_denoter():

    def __init__(self, rect_width:int=107, rect_height:int=48, car_park_positions_path:pickle="data/source/CarParkPos"):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.car_park_positions_path = car_park_positions_path
        self.car_park_positions = list()

    def read_positions(self)->list:
        """Lee el archivo pickle para evitar cualquier corrupción o error de datos.

        Devoluciones
        -------
        lista
            Lista de las tuplas que almacena las coordenadas del punto superior izquierdo del rectángulo del aparcamiento. Ejemplo de demostración: [(x_1, y_1), ..., [x_n, y_n]]
        """
        
        try:
            self.car_park_positions = pickle.load(open(self.car_park_positions_path, 'rb'))
        except Exception as e:
            print(f"Error: {e}\n It raised while reading the car park positions file.")

        return self.car_park_positions

    def mouseClick(self, events:int, x:int, y:int, flags:int, params:int):
        """Es la función de devolución de llamada para el evento de clic del mouse de acuerdo con la estructura opencv MouseCallBack.

        Parámetros
        ----------
        eventos: int
            una de las constantes cv2.MouseEventTypes
        x : entero
            La coordenada x del evento del ratón.
        y: int
           La coordenada y del evento del ratón.
        banderas: int
            una de las constantes cv2.MouseEventFlags.
        parámetros: int
            El parámetro opcional.
        """
        
        # añadir posición de aparcamiento a la lista
        if events==cv2.EVENT_LBUTTONDOWN:
            self.car_park_positions.append((x,y))
        
        # eliminar el aparcamiento correspondiente al clic del ratón
        if events==cv2.EVENT_MBUTTONDOWN:

           # averiguando y eliminando la etiqueta correspondiente.
            for index, pos in enumerate(self.car_park_positions):
                
                # unpacking
                x1,y1=pos
                
                # establecer la condición
                is_x_in_range= x1 <= x <= x1+self.rect_width
                is_y_in_range= y1 <= y <= y1+self.rect_height

              # comprobando que la etiqueta está en el rango
                if is_x_in_range and is_y_in_range:
                    self.car_park_positions.pop(index)

        # escribir las coordenadas de la etiqueta en el archivo
        with open(self.car_park_positions_path,'wb') as f:
            pickle.dump(self.car_park_positions,f)
        