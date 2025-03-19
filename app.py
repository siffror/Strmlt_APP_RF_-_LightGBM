import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageEnhance

# === Ladda endast LightGBM-modellen ===
lgb_model = joblib.load("lightgbm_mnist.pkl")

st.title("🔢 MNIST Sifferklassificering med LightGBM")
st.write("Ladda upp en handritad siffra och låt **LightGBM** klassificera den!")

# === Ladda upp bild ===
uploaded_file = st.file_uploader("📤 Ladda upp en bild", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # 📌 1. Visa originalbild
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Originalbild", use_column_width=True)

        # 📌 2. Förbehandla bilden
        image = image.convert("L")  # Gråskala
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Öka kontrasten
        image = image.resize((28, 28))  # Ändra storlek till 28x28 pixlar
        image = np.array(image, dtype=np.float32)  # Omvandla till numpy-array
        image = image / 255.0  # Normalisera pixelvärden till 0-1
        image = image.reshape(1, -1)  # Platta ut till (1, 784)

        # 📌 3. Gör prediktion med LightGBM
        pred_lgb = lgb_model.predict(image.astype(np.float32))[0]

        # 📌 4. Klassificeringsresultat
        st.markdown("---")
        st.subheader("📊 Klassificeringsresultat:")
        st.write(f"**🔵 LightGBMs Prediktion:** {pred_lgb}")

        # 📌 5. Visa resultat
        st.success(f"✅ Modellen klassificerade siffran som **{pred_lgb}**!")

    except Exception as e:
        st.error(f"🚨 Ett fel uppstod vid bildhantering: {e}")

else:
    st.warning("⚠️ Ladda upp en bild för att göra en klassificering.")
