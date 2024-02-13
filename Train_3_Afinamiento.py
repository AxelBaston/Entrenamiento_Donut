# Prueba de concepto para modelo de OCR

# Este código contiene toda la configuración que se realizará para el ajuste del modelo.

from torch.utils.data import DataLoader
import pandas as pd

import re
from nltk import edit_distance
import torch


configuracion = {
    "max_epochs": 30,
    "val_check_interval": 0.2,  # how many times we want to validate during an epoch
    "check_val_every_n_epoch": 1,
    "gradient_clip_val": 1.0,
    "num_training_samples_per_epoch": 800,
    "lr": 3e-5,
    "train_batch_sizes": [8],
    "val_batch_sizes": [1],
    # "seed":2022,
    "num_nodes": 1,
    "warmup_steps": 300,  # 800/8*30/10, 10%
    "result_path": "./result",
    "verbose": True,
}


class Entrenador:

    def __init__(self, t_data, v_data, batch_size, num_workers, modelo, procesador, device, config=configuracion, max_length=768, personal_record=0.05):
        self.train_data = Dataloader(t_data, training=True, batch_size=batch_size, num_workers=num_workers)
        self.valid_data = Dataloader(v_data, training=False, batch_size=batch_size, num_workers=num_workers)
        self.modelo = modelo
        self.procesador = procesador
        self.device = device
        self.configuracion = config
        self.optimizer = Configuracion_Optimizer(modelo=modelo, config=self.configuracion)
        self.hist_train = pd.DataFrame(columns=["Epoca", "Batch ID", "Loss"])  # Inicializar como DataFrame vacío
        self.hist_val = pd.DataFrame(columns=["Epoca", "Batch ID", "Loss"])  # Inicializar como DataFrame vacío
        self.max_length = max_length
        self.personal_record = personal_record

    def Entrenamiento(self):
        # Activar el modo de entrenamiento del modelo.
        self.modelo.train()
        # Ciclo para las epocas.
        for epoch in range(self.configuracion.get("max_epochs")):
            # Entrenamiento
            for batch_id, batch in enumerate(self.train_data):
                # Llamada a la función Training_step
                loss = self.Training_step(batch, batch_id, epoch)

                # Backpropagation y actualización de los pesos del modelo
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Imprimir información de la validación
            for batch_info in self.hist_train:
                print(f"Batch ID: {batch_info[0]}, Loss: {batch_info[1]}")

            # Validación
            for batch_idx, batch in enumerate(self.valid_data):
                # Llamada a la función Validation_step
                scores = self.Validation_step(batch, batch_idx, epoch)
            # Inicializar la puntuación del modelo.
            puntuacion_validacion = sum(scores)
            # Imprimir información de la validación
            for batch_info in self.valid_data:
                print(f"Batch Index: {batch_info[0]}, Mean Score: {batch_info[1]}")

            # Guardar los pesos del nuevo modelo.
            if puntuacion_validacion < self.mejor_puntuacion_validacion:
                self.mejor_puntuacion_validacion = puntuacion_validacion
                torch.save(self.modelo.state_dict(), 'mejor_modelo.pth')

    def Training_step(self, batch, batch_id, epoca):
        # Extraer los elementos del batch
        pixel_values = torch.stack(batch[0])  # Convertir todas las imágenes del batch en un tensor de 4 dimensiones
        labels = torch.stack(batch[1])  # Convertir todas las etiquetas del batch en un tensor
        # Generar una instancia de entrenamiento
        outputs = self.modelo(pixel_values, labels=labels)
        # Calcular la pérdida del entrenamiento para esta instancia.
        loss = outputs.loss

        # Guardar la información para el análisis del entrenamiento.
        registro = {"Época": epoca, "Batch ID": batch_id, "Loss": loss}
        self.hist_train = pd.concat([self.hist_train, pd.DataFrame(registro, index=[0])], ignore_index=True)

        print(f'Valor de pérdida:{loss}')
        return loss

    def Validation_step(self, batch, batch_idx, epoca):
        # Extraer los elementos del batch
        pixel_values = torch.stack(batch[0])  # Convertir todas las imágenes del batch en un tensor de 4 dimensiones

        labels = torch.stack(batch[1])  # Convertir todas las etiquetas del batch en un tensor

        answers = batch[2]
        # Mostrar cuál es la dimensión del batch.
        batch_size = pixel_values.shape[0]

        # Inicializar el modelo para las imágenes de entrada.
        decoder_input_ids = torch.full((batch_size, 1), self.modelo.config.decoder_start_token_id, device=self.device)
        # Generar la predicción con los parámetros (Ver documentación para conocer los parámetros).
        outputs = self.modelo.generate(pixel_values,
                                 decoder_input_ids=decoder_input_ids,
                                 max_length=self.max_length,
                                 early_stopping=True,
                                 pad_token_id=self.procesador.tokenizer.pad_token_id,
                                 eos_token_id=self.procesador.tokenizer.eos_token_id,
                                 use_cache=True,
                                 num_beams=1,
                                 # En este caso no buscamos entregar muchas respuestas para un solo caso. Por lo que se mantendra en 1.
                                 bad_words_ids=[[self.procesador.tokenizer.unk_token_id]],
                                 return_dict_in_generate=True, )
        # Inicializar las predicciones.
        predictions = []

        # Iteramos sobre los tokens generados por el modelo.
        for seq in self.procesador.tokenizer.batch_decode(outputs.sequences):
            # Eliminar los tokens de end of sentence y el los tokens de relleno (pad token)
            seq = seq.replace(self.procesador.tokenizer.eos_token, "").replace(self.procesador.tokenizer.pad_token, "")
            # Eliminar el token de inicio de tarea
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            # Añadirmos la predicción a un vector
            predictions.append(seq)

        scores = []
        # Iteramos entre las predicciones y las respuestas.
        for pred, answer in zip(predictions, answers):
            # Eliminar los tokens especiales
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", answer, count=1)
            # Eliminar los end of sentence token de la respuesta
            answer = answer.replace(self.procesador.tokenizer.eos_token, "")
            # Calcular el valor de perdida según la distancia de Levenstein (Documentación) normalizada
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            # Si es necesario mostrar los resultados.
            if self.configuracion.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f" Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        # Guardar valores para el análisis de rendimiento
        registro = {"Época": epoca, "Batch Index": batch_idx, "Mean Score": scores[0]}
        self.hist_val = pd.concat([self.hist_val, pd.DataFrame(registro, index=[0])], ignore_index=True)

        return scores




# Define tu función de collate personalizada
def my_collate(batch):
    tensors1, tensors2, diccionarios = zip(*batch)
    return tensors1, tensors2, diccionarios


def Dataloader(dataset, training=True, batch_size=3, num_workers=4):
    dataset_loaded = DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=num_workers, collate_fn=my_collate)
    return dataset_loaded

def Configuracion_Optimizer(modelo, config= configuracion):
    # you could also add a learning rate scheduler if you want
    optimizer = torch.optim.Adam(modelo.parameters(), lr=config.get("lr"))
    return optimizer



class Entreneator:
    def __init__(self, modelo, optimizador, criterio):
        self.modelo = modelo
        self.optimizador = optimizador
        self.criterio = criterio
        self.mejor_puntuacion_validacion = 0.5 # Inicializamos con un valor muy grande

    def entrenar(self, data_loader, epochs=1, device='cpu'):
        self.modelo.to(device)
        for epoch in range(epochs):
            self.modelo.train()
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                self.optimizador.zero_grad()
                outputs = self.modelo(inputs)
                loss = self.criterio(outputs, targets)
                loss.backward()
                self.optimizador.step()
                total_loss += loss.item()
                del inputs, targets, outputs  # Liberar memoria explícitamente
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader)}')

            # Validar y guardar el mejor modelo
            puntuacion_validacion = self.validar(data_loader, device)


    def validar(self, data_loader, device='cpu'):
        self.modelo.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.modelo(inputs)
                loss = self.criterio(outputs, targets)
                total_loss += loss.item()
        puntuacion_validacion = total_loss / len(data_loader)
        print(f'Loss en conjunto de validación: {puntuacion_validacion}')
        return puntuacion_validacion