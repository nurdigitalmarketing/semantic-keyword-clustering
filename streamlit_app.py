import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import nltk
import string
from io import BytesIO

# Assicurati di aver scaricato i pacchetti NLTK necessari
nltk.download('stopwords')
nltk.download('punkt')

# Funzione di preprocessamento
def preprocess(text, lang):
    stop_words = set(stopwords.words(lang))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Funzione per il clustering e la creazione dei risultati
def cluster_keywords(data, lang, num_clusters):
    data['processed_keywords'] = data['keywords'].apply(lambda x: preprocess(x, lang))
    sentences = [row.split() for row in data['processed_keywords']]
    w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = [sum([w2v_model.wv[word] for word in sentence]) / len(sentence) for sentence in sentences]
    
    if len(word_vectors) < num_clusters:
        raise ValueError(f"Number of samples ({len(word_vectors)}) must be >= number of clusters ({num_clusters}).")
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(word_vectors)
    data['cluster'] = kmeans.labels_
    return data

# Titolo dell'applicazione
st.title('Semantic Keyword Clustering')

# Selezione della lingua
lang = st.selectbox('Select language for stopwords', ['english', 'spanish', 'french', 'german', 'italian'])

# Input delle parole chiave o caricamento del file CSV
input_option = st.radio("Choose input method", ('Manual Input', 'Upload CSV'))

data = None
if input_option == 'Manual Input':
    keywords = st.text_area("Enter keywords separated by commas")
    if keywords:
        data = pd.DataFrame({'keywords': keywords.split(',')})
elif input_option == 'Upload CSV':
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if 'keywords' not in data.columns:
            st.error('The CSV file must contain a column named "keywords".')
            data = None

# Selezione del numero di cluster
if data is not None:
    num_samples = len(data)
    st.write(f"Number of samples: {num_samples}")
    num_clusters = st.slider('Select number of clusters', 1, min(20, num_samples), 5)

    # Esegui il clustering
    if st.button('Cluster Keywords') and num_samples >= num_clusters:
        try:
            clustered_data = cluster_keywords(data, lang, num_clusters)
            # Mostra i risultati
            st.write(clustered_data)

            # Esportazione dei risultati
            output_format = st.selectbox('Select export format', ['CSV', 'Excel'])

            if output_format == 'CSV':
                csv = clustered_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download clustered keywords as CSV",
                    data=csv,
                    file_name='clustered_keywords.csv',
                    mime='text/csv',
                )
            elif output_format == 'Excel':
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    clustered_data.to_excel(writer, index=False, sheet_name='Sheet1')
                st.download_button(
                    label="Download clustered keywords as Excel",
                    data=output.getvalue(),
                    file_name='clustered_keywords.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
        except ValueError as e:
            st.error(e)
    elif num_samples < num_clusters:
        st.error(f"Number of samples ({num_samples}) must be >= number of clusters ({num_clusters}).")

# Avvia l'app con: streamlit run app.py
