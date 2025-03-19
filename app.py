import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import gdown

# Google Drive fil-ID för din modell
file_id = '194s6DLo76VPeuDOCDceNaq84B39tEU8a'  # Fil-ID från din Google Drive-länk
url = f'https://drive.google.com/uc?export=download&id={file_id}'

# Spara modellen från Google Drive
output = 'my_trained_model.h5'

# Ladda ner modellen
gdown.download(url, output, quiet=False)

# Ladda modellen
model = load_model(output)

# Nu kan du använda modellen för att göra förutsägelser i din app

# Titeln på appen
st.title("Handskriven Sifferigenkänning med CNN")
st.write("Rita en siffra i fönstret nedan och modellen kommer att förutsäga vilken siffra det är.")

# Rita en siffra på canvas
canvas_result = st_canvas(
    fill_color="black",  # Fyllningsfärg
    stroke_width=20,  # Tjockleken på pennan
    stroke_color="white",  # Pennans färg
    background_color="black",  # Bakgrundsfärg
    height=280,  # Höjd på canvas
    width=280,  # Bredd på canvas
    drawing_mode="freedraw",  # Låt användaren rita fritt
    key="canvas"
)

# När användaren ritar en siffra, gör en förutsägelse
if canvas_result.image_data is not None:
    # Konvertera den ritade bilden till en PIL-bild
    image = Image.fromarray(canvas_result.image_data.astype("uint8"))
    
    # Förbehandla bilden
    image = image.convert("L")  # Konvertera till gråskala
    image = image.resize((28, 28))  # Ändra storlek till 28x28
    image_array = np.array(image)  # Omvandla till array
    image_array = image_array.reshape(1, 28, 28, 1)  # Omforma till (1, 28, 28, 1)
    image_array = image_array.astype("float32") / 255.0  # Normalisera pixelvärden
    
    # Gör en förutsägelse med modellen
    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction, axis=1)[0]
    
    # Visa resultatet
    st.write(f"Modellen förutspår att detta är siffran: {predicted_label}")
