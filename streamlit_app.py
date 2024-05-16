import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st
import io

# Titolo dell'app Streamlit
st.title('Keyword Clustering')

# Opzione per inserire le keyword a mano o caricare un CSV
input_option = st.selectbox("Seleziona il metodo di input:", ["Carica CSV", "Inserisci manualmente"])

if input_option == "Carica CSV":
    uploaded_file = st.file_uploader("Scegli un file CSV", type="csv")

    if uploaded_file is not None:
        # Carica i dati dal CSV
        data = pd.read_csv(uploaded_file)
        st.write(data)
else:
    keyword_input = st.text_area("Inserisci le keyword (una per riga):")
    
    if keyword_input:
        keywords = keyword_input.split("\n")
        data = pd.DataFrame(keywords, columns=["keywords"])
        st.write(data)

# Seleziona la lingua delle stop-words
stop_words_lang = st.selectbox("Seleziona la lingua delle stop-words:", ["english", "italian", "french", "spanish", "german"])

if 'data' in locals():
    # Vectorizza le keyword
    vectorizer = TfidfVectorizer(stop_words=stop_words_lang)
    X = vectorizer.fit_transform(data['keywords'])

    # Applica il clustering KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)

    # Aggiungi le etichette dei cluster ai dati
    data['cluster'] = kmeans.labels_

    # Mostra il dataframe con le etichette dei cluster
    st.write(data)

    # Traccia i risultati
    plt.scatter(X[:, 0].toarray(), X[:, 1].toarray(), c=data['cluster'])
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Keyword Clustering')
    
    # Mostra il grafico in Streamlit
    st.pyplot(plt)

    # Esporta il documento finale
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Scarica il CSV con i cluster",
        data=csv,
        file_name='clustered_keywords.csv',
        mime='text/csv',
    )

# Istruzione per l'utente
st.text("Carica un file CSV o inserisci le keyword manualmente per vedere i risultati del clustering.")
