import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageEnhance

# === Ladda de sparade modellerna ===
rf_model = joblib.load("random_forest_mnist.pkl")
lgb_model = joblib.load("lightgbm_mnist.pkl")

st.title("🔢 MNIST Sifferklassificering med Streamlit")
st.write("Ladda upp en handritad siffra och låt **Random Forest** och **LightGBM** klassificera den!")

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

        # 📌 3. Visa bildens diagnostik
        st.markdown("### 🔍 Bildens diagnostik:")
        st.write(f"- **Min-värde:** {image.min():.4f}")
        st.write(f"- **Max-värde:** {image.max():.4f}")
        st.write(f"- **Medelvärde:** {image.mean():.4f}")
        st.write(f"- **Standardavvikelse:** {image.std():.4f}")
        st.write(f"- **Shape:** {image.shape}")

        # 📌 4. Gör prediktioner med båda modellerna
        pred_rf = rf_model.predict(image.astype(np.float32))[0]
        pred_lgb = lgb_model.predict(image.astype(np.float32))[0]

        # 📌 5. Klassificeringsresultat
        st.markdown("---")
        st.subheader("📊 Klassificeringsresultat:")

        st.write(f"**🟢 Random Forests Prediktion:** {pred_rf}")
        st.write(f"**🔵 LightGBMs Prediktion:** {pred_lgb}")

        # 📌 6. Kontrollera om resultaten matchar
        if pred_rf == pred_lgb:
            st.success(f"✅ Båda modellerna klassificerade siffran som **{pred_rf}**!")
        else:
            st.warning(f"⚠️ Modellerna gav olika resultat! RF: {pred_rf}, LightGBM: {pred_lgb}")

    except Exception as e:
        st.error(f"🚨 Ett fel uppstod vid bildhantering: {e}")

else:
    st.warning("⚠️ Ladda upp en bild för att göra en klassificering.")
