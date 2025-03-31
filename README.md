# Estimación de Velocidad utilizando Redes-Neuronales 🚀

Entrenamiento de la CNN YoloV5 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oRK8CG8riQmib0-v39mGP81-mGGeDpt8?usp=sharing)


¡Bienvenid@! Este proyecto utiliza Redes Neuronales Convolucionales para hacer estimaciones de velocidad de automóviles. :car: :motorway:

https://github.com/user-attachments/assets/50d8706c-bb59-4d8d-a29c-8b3c3b75c318

## Descripción  🚗💨
Este proyecto es un experimento práctico que aplica CNNs en el contexto del reconocimiento y seguimiento de vehículos para estimar su velocidad.

Utilizando un conjunto de datos de video de tráfico real tomado en el Boulevard Suyapa, en Tegucigalpa Honduras, se entrena una CNN para identificar vehículos y calcular su velocidad a partir de secuencias de imágenes. Este proceso involucra la detección de los autos en cada cuadro y el seguimiento de su movimiento a lo largo del tiempo, además se destaca la aplicación de una transformación geométrica de perspectiva en el proceso de cálculo de la velocidad. Este procedimiento implica la representación matricial de las imágenes capturadas por cámaras y la utilización de operaciones tensoriales para modelar los cambios de perspectiva conforme el automóvil se desplaza.

 ## Objetivos específicos del experimento :pencil2:

1. **Detección Automática de Vehículos:**
   - Identificar diversos tipos de vehículos mediante avanzadas técnicas de visión por computadora. <br/>
       <table>
           <tr>
             <td align="center">
               <img src="https://github.com/user-attachments/assets/3e6f9944-4126-4e8c-ae49-0dc011e3e309" width="300"/>
               <br/>
               <sub>Detección 1</sub>
             </td>
             <td align="center">
               <img src="https://github.com/user-attachments/assets/8d892014-0929-43f1-97e9-b4b0a2094785" width="300"/>
               <br/>
               <sub>Detección 2</sub>
             </td>
            <td align="center">
               <img src="https://github.com/user-attachments/assets/e7f36c8b-e52a-419d-bab2-10e94be074f7" width="300"/>
               <br/>
               <sub>Detección 3</sub>
             </td>
           </tr>
      </table>


2. **Seguimiento de Vehículos:**
   - Mantener un seguimiento continuo de los vehículos detectados para monitorear su trayectoria a lo largo del tiempo.
      <table>
           <tr>
             <td align="center">
               <img src="https://github.com/user-attachments/assets/1d4cc1fa-0af1-484e-b44d-19ffbf6ffc90" width="1000"/>
               <br/>
               <sub>Seguimiento de objetos a lo largo del tiempo mediante el uso de dataframes.</sub>
             </td>
      </table>

     

3. **Estimación de Velocidad:**
   - Calcular la velocidad de cada vehículo detectado y en seguimiento.
   - Utilizar información de la secuencia de imágenes o video para una estimación precisa.
  
  
  ## Construido con 🛠️

Este proyecto fue construido utilizando las siguientes herramientas:

- [Google Colab](https://colab.research.google.com/): Entorno de cuadernos colaborativos para la ejecución de código Python, especialmente útil para tareas de aprendizaje profundo.
- [PyTorch](https://pytorch.org/): Biblioteca de aprendizaje profundo de código abierto para Python.
- [NumPy](https://numpy.org/): Biblioteca para la manipulación eficiente de arreglos y matrices en Python.
- [OpenCV](https://opencv.org/): Biblioteca de visión por computadora y procesamiento de imágenes.
- [YOLOv5](https://github.com/ultralytics/yolov5): Implementación de YOLO (You Only Look Once), un modelo de detección de objetos en tiempo real.
- [Roboflow](https://roboflow.com/): Plataforma de preparación y gestión de datos para visión por computadora. Simplifica el flujo de trabajo de preparación de datos para modelos de aprendizaje profundo.

  
### Pre-requisitos 📋

Antes de comenzar, asegúrate de tener acceso a un entorno de Google Colab. Puedes acceder a Google Colab desde tu navegador web en [colab.research.google.com](https://colab.research.google.com/).

1. **Conexión a Google Drive (opcional):** Si planeas utilizar Google Drive para almacenar o acceder a datos, modelos, etc., es recomendable conectar Colab con Google Drive. Puedes hacerlo ejecutando el siguiente código en una celda de Colab:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Instalación de PyTorch:** PyTorch generalmente ya está preinstalado en Google Colab, pero puedes verificarlo e instalarlo si es necesario:

    ```python
    import torch
    print(torch.__version__)
    ```

    Si no está instalado, puedes hacerlo con:

    ```python
    !pip install torch torchvision
    ```

3. **Importación de Bibliotecas Comunes:**
    - NumPy:

    ```python
    import numpy as np
    ```

    - Matplotlib (para visualización):

    ```python
    import matplotlib.pyplot as plt
    ```

4. **Conexión a GPU (opcional):** Si deseas aprovechar una GPU en Colab, puedes hacerlo seleccionando `Entorno de ejecución > Cambiar tipo de entorno de ejecución` y eligiendo `Acelerador de hardware > GPU`.

Recuerda que Google Colab reinicia el entorno después de un período de inactividad, por lo que debes ejecutar estas configuraciones al principio de tu cuaderno o script en Colab.

## Uso de Roboflow en el Proyecto 🤖

```python
# Importa Roboflow
from roboflow import Roboflow

# Configura tu API Key
rf = Roboflow(api_key="TU_API_KEY")

# Selecciona Proyecto y Dataset
project = rf.workspace("NOMBRE_DE_TU_WORKSPACE").project("NOMBRE_DE_TU_PROYECTO")
dataset = project.version(1).download("yolov5")

# Descarga el Modelo (Ejemplo con YOLOv5)
dataset = project.version(1).download("yolov5")
```



## Autores ✒️

* **Henry Ocampo** - *Investigación y asesor* - [henryocampo](#henry-ocampo)
* **Iván Henríquez** - *Obtención de data* - [ivanhenriquez](#ivan-henriquez)
* **María Fernanda Mendoza** - *Diseño y creación de base de datos para entrenamiento* - [mfernandamendoza](#maria-fernanda-mendoza)
* **Ruth Moreno** - *Modelo de detección* - [ruthmoreno](#ruth-moreno)


