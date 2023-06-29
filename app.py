import streamlit as st
import pickle
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

def main():
    st.title("Input Kalimat")
    st.write("Masukkan kalimat di bawah ini:")
    with open('new_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Load the KNN model
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)

    kalimat = st.text_input("Kalimat:")

    if st.button("Prediksi"):
        # Lakukan proses prediksi di sini
        test_input = vectorizer.transform([kalimat])
        predictions = knn_model.predict(test_input)
        labels = ["negatif", "positif"]
        st.write("""## Sentimen ini bernilai """, labels[predictions[0]])





if __name__ == "__main__":
    main()