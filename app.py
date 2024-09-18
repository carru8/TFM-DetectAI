import gradio as gr  # type: ignore
import tensorflow as tf  # type: ignore
from PIL import Image
import numpy as np

# Cargar el modelo previamente entrenado
modelo = tf.keras.models.load_model('combinado.h5')

# Definir la función de predicción
def predict(image):
    # Redimensionar la imagen a 32x32 píxeles
    image = image.resize((32, 32))  
    
    # Convertir la imagen a un array de numpy y normalizarla
    img_ar = np.asarray(image) / 255.0  # Normalizar la imagen
    img_ar_rs1 = img_ar.reshape(-1, 32, 32, 3)  # Redimensionar la imagen al tamaño esperado por el modelo

    # Realizar la predicción
    pred = modelo.predict(img_ar_rs1)[0][0]
    
    # Convertir la probabilidad a porcentaje
    porcentaje = pred * 100

    # Interpretar la predicción
    if pred > 0.6:
        return f"La imagen es Real con un {porcentaje:.2f}% de confianza."
    else:
        return f"La imagen está generada por IA con un {100 - porcentaje:.2f}% de confianza."

# Crear la interfaz de Gradio mejorada
interfaz = gr.Interface(
    fn=predict,  # Función de predicción
    inputs=gr.Image(type="pil", label="Carga tu imagen"),  # Input de tipo imagen con herramientas de edición
    outputs=gr.Textbox(label="Resultado de la predicción"),  # Output con un label más claro
    title="DetectAI ",  # Título de la aplicación
    description="Una aplicación que identifica las imágenes generadas por Inteligencia Artificial.",  # Descripción de la aplicación
    examples=["Jarulis.jpg"],  # Imagen de ejemplo para cargar
    theme="compact",  # Tema compacto para una apariencia más elegante
    allow_flagging="never",  # Desactiva la opción de flagging
    live=False,  # Desactiva el modo en vivo para optimizar el rendimiento
)

# Iniciar la aplicación
if __name__ == "__main__":
    interfaz.launch(share=True)

    
    

