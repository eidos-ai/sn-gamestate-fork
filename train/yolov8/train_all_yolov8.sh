#!/bin/bash

# Lista de variantes de YOLOv8
variants=("yolov8x.pt")

# Configuración común
yaml_config="yolo_config.yaml"
mlflow_flag="--mlflow"

# Bucle para ejecutar todas las variantes
for variant in "${variants[@]}"; do
    echo "Entrenando con la variante: $variant"
    python train_yolov8.py -m "$variant" -y "$yaml_config" -e 20 "$mlflow_flag"
    
    # Verificar si el entrenamiento fue exitoso
    if [ $? -eq 0 ]; then
        echo "Entrenamiento con $variant completado con éxito."
    else
        echo "Error durante el entrenamiento con $variant."
        exit 1  # Detener el script si hay un error
    fi
done

echo "Todos los entrenamientos han finalizado."