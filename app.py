import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageEnhance
from streamlit_drawable_canvas import st_canvas

# === Ladda LightGBM-modellen (Cachas för snabbare inferens) ===
@st.cache_resource
def load_model():
    """Laddar modellen en gång och cachar den."""
    return joblib.load("lightgbm_mnist.pkl")

lgb_model = load_model()

st.title("🖌️ MNIST Sifferklassificering - Rita eller ladda upp en bild!")

# === Välj inmatningsmetod ===
option = st.radio("Välj inmatningsmetod:", ("✏️ Rita en siffra", "📤 Ladda upp en bild"))

# === Om användaren väljer att rita ===
if option == "✏️ Rita en siffra":
    st.sidebar.header("✏️ Ritverktyg")
    stroke_width = st.sidebar.slider("Pennans tjocklek:", 1, 25, 10)
    bg_color = st.sidebar.color_picker("Bakgrundsfärg", "#000000")  # Svart bakgrund
    drawing_mode = st.sidebar.selectbox("Ritläge", ("freedraw", "line"))

    # 📌 Skapa ritcanvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # Transparent
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",  # Vit penna
        background_color=bg_color,
        width=280,  # 10x större än MNIST (för bättre upplösning)
        height=280,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # === Bearbeta den ritade siffran ===
    if canvas_result.image_data is not None:
        # 📌 1. Konvertera bilden till PIL-format
        image = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")

        # 📌 2. Konvertera till gråskala och justera kontrasten
        image = image.convert("L")  # Gråskala
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Öka kontrasten

        # 📌 3. Ändra storlek till 28x28 (MNIST-format)
        image = image.resize((28, 28))

        # 📌 4. Normalisera bilden (0-1) och platta ut till 1D-vektor
        image = np.array(image, dtype=np.float32) / 255.0
        image = image.reshape(1, -1)

        # 📌 5. Visa den förbehandlade bilden
        st.image(image.reshape(28, 28), caption="🖼 Förbehandlad bild", width=100)

        # === Gör prediktion med LightGBM ===
        pred_lgb = lgb_model.predict(image.astype(np.float32))[0]

        # 📌 6. Klassificeringsresultat
        st.markdown("---")
        st.subheader("📊 Klassificeringsresultat:")
        st.write(f"**🔵 LightGBMs Prediktion:** {pred_lgb}")
        st.success(f"✅ Modellen klassificerade siffran som **{pred_lgb}**!")

    else:
        st.warning("✏️ Rita en siffra på canvasen för att klassificera den!")

# === Om användaren väljer att ladda upp en bild ===
elif option == "📤 Ladda upp en bild":
    uploaded_file = st.file_uploader("Ladda upp en bild", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # 📌 1. Visa originalbild
            image = Image.open(uploaded_file)
            st.image(image, caption="🖼 Originalbild", use_column_width=True)

            # 📌 2. Förbehandla bilden
            image = image.convert("L")  # Gråskala
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # Öka kontrasten
            image = image.resize((28, 28))
            image = np.array(image, dtype=np.float32) / 255.0  # Normalisera
            image = image.reshape(1, -1)

            # 📌 3. Gör prediktion
            pred_lgb = lgb_model.predict(image.astype(np.float32))[0]

            # 📌 4. Klassificeringsresultat
            st.markdown("---")
            st.subheader("📊 Klassificeringsresultat:")
            st.write(f"**🔵 LightGBMs Prediktion:** {pred_lgb}")
            st.success(f"✅ Modellen klassificerade siffran som **{pred_lgb}**!")

        except Exception as e:
            st.error(f"🚨 Ett fel uppstod: {e}")

    else:
        st.warning("⚠️ Ladda upp en bild för att göra en klassificering.")
