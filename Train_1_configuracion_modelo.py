# Prueba de concepto para modelo de OCR
​
# El modelo afinado se encuentra en: https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2/tree/main
​
### Extracción y configuración del modelo.
​
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
​
import torch
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig, DonutProcessor
​
from Train_1_json_to_token import MiDataset
​
from Train_2_Ajuste import Entrenador
​
from datasets import load_dataset
# Código base para el entrenamiento del modelo.
​
​
# Ruta de la configuración
ruta_config = '../Archivos adicionales/donut-base-finetuned-cord-v2/config.json'
​
# Ruta del afinamiento del modelo
ruta_modelo = '../Archivos adicionales/donut-base-finetuned-cord-v2/donut_pytorch_model.bin'
​
# Ruta al directorio que contiene los archivos del Procesador
procesador_dir = "../Archivos adicionales/donut-base-finetuned-cord-v2/"
​
​
​
# Cargar el procesador. Recordemos que Donut es un modelo de procesamiento de imágenes y también tiene un tokenizador para generación de texto
procesador = DonutProcessor.from_pretrained(procesador_dir)
​
# Cargar la configuración del modelo desde el archivo JSON
configuracion = VisionEncoderDecoderConfig.from_json_file(ruta_config)
​
​
​
# Importar el modelo
​
modelo_donut = VisionEncoderDecoderModel(configuracion)
​
# Cargar los datos pre-entrenados del modelo
modelo_donut.load_state_dict(torch.load(ruta_modelo, map_location=torch.device('cpu')))
​
# Ajustar la matriz de embeddings del decodificador para incluir el nuevo token
modelo_donut.decoder.resize_token_embeddings(len(procesador.tokenizer))
​
## Obtener los datos:
​
# Cargar el dataset de un repositorio en git
dataset = load_dataset("naver-clova-ix/cord-v2")
​
train_dataset = dataset['train']
​
val_dataset = dataset['validation']
​
​
​
# Extraer el dataset de entrenamiento.
entrenamiento = MiDataset(train_dataset, modelo= modelo_donut, training = True, procesor = procesador, max_length=768)
​
validacion = MiDataset(val_dataset, modelo= modelo_donut, training = True, procesor = procesador, max_length=768)
​
# Liberar memoria para la optimización.
del dataset, train_dataset, val_dataset
​
# Cambiar el pad id del modelo.
modelo_donut.config.pad_token_id = procesador.tokenizer.pad_token_id
# Cambiar el decoder start token del modelo.
modelo_donut.config.decoder_start_token_id = procesador.tokenizer.convert_tokens_to_ids(['<s_cord-v2>'])[0]
​
# Inicializar el device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
​
batch_size = 2
​
num_workers = 5
​
# Crear una instancia del Entrenador
entrenador = Entrenador(t_data=entrenamiento, v_data=validacion, batch_size=batch_size, num_workers=num_workers, modelo=modelo_donut, procesador=procesador, device=device)
​