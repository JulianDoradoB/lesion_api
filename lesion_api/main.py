from fastapi import FastAPI, File, UploadFile
from transformers import BeitForImageClassification, BeitImageProcessor
from PIL import Image
import torch
import io

app = FastAPI()

model = BeitForImageClassification.from_pretrained(
    "ALM-AHME/beit-large-patch16-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20"
)
processor = BeitImageProcessor.from_pretrained(
    "ALM-AHME/beit-large-patch16-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20"
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]

    # Diccionario de descripciones
    descripciones = {
        "bcc": "Carcinoma basocelular: un tipo de cáncer de piel de crecimiento lento...",
        "nv": "Nevus melanocítico: también conocido como lunar común...",
        "mel": "Melanoma: tipo de cáncer de piel más grave...",
        "akiec": "Queratosis actínica...",
        "df": "Dermatofibroma...",
        "vasc": "Lesión vascular...",
        "bkl": "Léntigo solar...",
    }

    # Obtener descripción correspondiente
    descripcion = descripciones.get(predicted_label.lower(), "Descripción no disponible.")

    return {
        "diagnosis": predicted_label,
        "description": descripcion
    }
