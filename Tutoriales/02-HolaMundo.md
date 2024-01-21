# Hola Mundo Quantico

Para realizar lo especificado a continuacion puedes clonar el repositorio actual. El repositorio cuenta con dos carpetas: 
* __IntroQC__, que contiene a su vez la carpeta '_QuantumCUDA_' con un archivo _.zip_ con los script de python que utilizaremos y '_quantum.yml_' que nos servira para crear nuestro entorno. Ademas contiene un archivo '_qcCUDA.zip_' que vamos a ignorar.
* __Tutoriales__, una carpeta que contiene los pasos a seguir para hacer nuestro primer 'Hola mundo cuantico!'.


## Creacion del Entorno
Ahora, ya en nuestro servidor remoto, vamos a seguir los siguientes pasos:

1. Cargar el modulo de conda :```module load conda```.

1. Crear el entorno: `conda env create -f quantum.yml`. 
Considere aquí que el archivo __.yml__ que se encuentra en la carpeta _QuantumCUDA_. (Considere que para crear el entorno se debe encontrar en la carpeta indicada)

> Utilice ``pwd``para saber su ubicacion exacta en el servidor

3. Ya creado, tendremos los siguientes comandos para activar o desactivar nuestro entorno: `conda activate quantum` o `conda deactivate`. Considere que el entorno tomara el nombre que se haya especificado en el archivo _.yml_ en la seccion _name_.

>[!NOTE]
> Si desea eliminar el entorno, utilice
> `conda env remove --name quantum`

### Agregar Librerias
Con el entorno ya creado, ahora vamos a agregar algunas de las librerias de _CUDA_ necesarias.

>[!IMPORTANT]
>
> Aplique los siguientes comandos con el entorno activado. 
> ``pip install cuda-quantum cuda-python cuquantum-cu11 custatevec-cu11 cutensor-cu11 cutensornet-cu11``
> ``conda install -c nvidia cuda-python``

Para revisar si toda las librerias estan instaladas puedes usar ``conda list``. 

## Usar Archivos
Estando el la carpeta _QuantumCUDA_ utilice ``unzip CuQuantum.zip`` , se descimprimirá el archivo y obtendrá todos los script de python junto con algunos de CUDA.

* Para correr python desde la terminal, activamos primero nuestro entorno y luego ejecutamos ```python3 'nombre-archivo'.py```.

* Para correr CUDA haremos lo siguiente: 
        1. ``module load cuda/11.7 `` (_YUPANA_)
        2. ``` nvcc -o mi_programa 'nombre-archivo'.cu -lcustatevec ```
        3. ```./miprograma```

