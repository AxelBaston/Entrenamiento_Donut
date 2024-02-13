# Prueba de concepto para modelo de OCR

# Esta código tiene la finalidad de convertir un diccionario de imágen a los tokens que nuestro modelo consume además de añadir tokens separadores para el entendimiento del documento.

# Los elementos del conjunto de datos se eseparan en dos.
## {'image':} Toda la meta información de la imágen
## {'ground_truth':}  El texto contenido en la imágen en forma de diccionario. Además toda la metadata (No se a que se refiera, pero no es necesaria para el entrenamiento)
# El hipervínculo del conjunto de datos utilizado para esta prueba de concepto es : https://huggingface.co/datasets/naver-clova-ix/cord-v2

import json
import random
import torch
from torch.utils.data import Dataset


# Define tu conjunto de datos personalizado
class MiDataset(Dataset):
    def __init__(self, data, modelo, procesor, training=True, max_length=768):
        self.data = data
        self.modelo = modelo
        self.procesor = procesor
        self.training = training
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        muestra = self.data[index]
        ground_truth = json.loads(muestra['ground_truth'])

        if 'gt_parses' in ground_truth:
            gt_jsons = ground_truth['gt_parses']
        else:
            gt_jsons = ground_truth.get('gt_parse', None)
            if not isinstance(gt_jsons, list) and not isinstance(gt_jsons, dict):
                print("El siguiente elemento es detectado como string:")
                print(gt_jsons)
                return None  # Skip this sample if ground truth is not in the expected format

        if isinstance(gt_jsons, dict):
            elemento, self.procesor = transformar_diccionario(diccionario=gt_jsons, add_tokens=self.training, procesor=self.procesor)
        else:
            return None  # Skip this sample if ground truth is not in the expected format

        self.modelo.decoder.resize_token_embeddings(len(self.procesor.tokenizer))

        tokens = Targets_calculator(elemento=[elemento], procesor=self.procesor, max_length=self.max_length, training=self.training)

        image = Imagen_convertor(muestra['image'], procesor=self.procesor, training=self.training)

        return image, tokens, elemento

def transformar_diccionario(diccionario, add_tokens, procesor):
    '''
    Función que se encarga de traducir un diccionario json en elementos que el modelo es capaz de comprender y seprar.
    Principalmente se utilizará para que el modelo comprenda la estructura de los datos en el entrenamiento.
    :param diccionario: Diccionario json del cual extrearemos la información de nuestro conjunto de datos.
    :param add_tokens: Decidir si es necesario añadir tokens al tokenizador
    :param tokenizer: El tokenizador del modelo
    :return: El diccionario listo para que el modelo lo convierta en sus ids y un tokenizador acutalizado con los parámetros añadidos.
    '''
    # Inicializamos la lista de partes de la cadena
    partes_cadena = []

    # Iterar sobre una clave y los valores de esta clave del diccionario.
    for clave, valor in diccionario.items():
        # Abrimos una clave para el diccionario.
        partes_cadena.append("<s_{}>".format(clave))

        # Si el valor sigue siendo un diccionario abrimos una nueva clave repitiendo la función
        if isinstance(valor, dict):
            # Ajustar el tokenizer y extraer el diccionario interno
            elemento, procesor = transformar_diccionario(valor, add_tokens=add_tokens, procesor=procesor)
            # Agregar la cadena al resultado
            partes_cadena.append(elemento)

        # Si el valor es una lista nos moveremos por los elementos de la lista (CUIDADO AQUÍ)
        elif isinstance(valor, list):
            # Cuando hay una lista en un diccionario. Ver documentación.
            indice = 0
            # Para cada elemento de la lista colocaremos su respectiva etiqueta al inicio y al final
            for item in valor:
                # Iteramos sobre las claves y valores de los elementos de la lista
                if isinstance(item, dict):
                    for subclave, subvalor in item.items():  # Iterar sobre las subclaves y subvalores del elemento
                        # Agregar elementos de apertura y cierre para las subclaves.
                        partes_cadena.append("<s_{}> {} </s_{}>".format(subclave, subvalor.replace(",", "") if isinstance(subvalor,str) else subvalor,subclave))
                        # Añadir los tokens al tokenizador
                        if add_tokens:
                            procesor.tokenizer.add_tokens(["</s_{}>".format(subclave), "<s_{}>".format(subclave)])
                elif isinstance(item, str):
                    if indice != len(valor)-1:
                        # Agregar el valor al resultado, eliminando comas si es una cadena
                        partes_cadena.append(" {}, ".format(item.replace(",", "") if isinstance(item, str) else valor))
                    else:
                        partes_cadena.append(" {} ".format(item.replace(",", "") if isinstance(item, str) else valor))
                    indice += 1
        # Si el valor es un valor escalar (cadena, número, etc.)
        else:
            # Agregar el valor a las partes de la cadena, eliminando comas si es una cadena
            partes_cadena.append(" {} ".format(valor.replace(",", "") if isinstance(valor, str) else valor))

        # Agregar la etiqueta de cierre con el nombre de la clave
        partes_cadena.append("</s_{}>".format(clave))

        # Si queremos añadir tokens:
        if add_tokens:
            # Añadir tokens al tokenizador
            procesor.tokenizer.add_tokens(["</s_{}>".format(clave), "<s_{}>".format(clave)])

    # Unir las partes de la cadena para formar el resultado final
    resultado = ''.join(partes_cadena)

    return resultado, procesor


def Imagen_convertor(sample, procesor, training=True):
    '''
    Convertir una imagen a un tensor de pytorch que el modelo pueda entender.
    :param sample: Imagen que se desea convertir.
    :return:
    '''

    # Convertir imágenes en pixeles y enmascarar algunas partes de la imágen cuando el modelo se esté entrenando
    # La última parte 'pt' es para indicar que se debe de colocar un formato pytorch al tensor de salida.
    pixel_values = procesor(sample, random_padding=training, return_tensors="pt").pixel_values

    # Si se devuelve un tensor de tamaño 3, no es necesario hacer un squeeze
    if pixel_values.ndim == 3:
        return pixel_values

    # Si se devuelve un tensor de tamaño 4, hacer un squeeze para eliminar la dimensión adicional
    elif pixel_values.ndim == 4:
        pixel_values = pixel_values.squeeze()

    return pixel_values


def Targets_calculator(elemento, procesor, max_length, training=True) -> torch.Tensor:
    '''
    Función utilizada mayormente para el entrenamiento.
    Esta función toma una de las secuencias de manera aleatoria de nuestro token para que el modelo se entrene con esta secuencia de tokens.
    Esto se suele hacer para prevenir el sobre ajuste y hacer un modelo más robusto.
    :param elemento: Oración aún no tokenizada en una lista.
    :param procesor:
    :param max_length:
    :param training: Indicar si el conjunto de datos es de entrenamiento.
    :return: Una o varias oraciones extraidas de los tokens.
    '''
    # Seleccionar una secuencia aleatoria
    target_sequence = random.choice(elemento)

    # Tokenizar los targets del modelo y transformarlo en un tensor de Pytorch
    input_ids = procesor.tokenizer(
        target_sequence,
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)  # Asegurar que el tensor esté en la GPU si es posible

    # Clonar los input_ids para poder usarlos sin necesidad de que el modelo sea llamado.
    labels = input_ids.clone()

    # Establecer como -100 al token pad para que el modelo no lo tome en cuenta en su entrenamiento
    labels[labels == procesor.tokenizer.pad_token_id] = -100  # model doesn't need to predict pad token

    return labels
