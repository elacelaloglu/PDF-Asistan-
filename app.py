import streamlit as st
import os
import tempfile
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- AYARLAR ---
# GROQ ANAHTARINI BURAYA YAPIŞTIR
GROQ_API_KEY = "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"

st.set_page_config(page_title="PDF Asistanı", layout="wide")
st.title("☁️ Canlı PDF Asistanı")
st.markdown("Sol taraftan bir PDF yükleyin ve hemen soru sormaya başlayın!")

# Yan Menü - Dosya Yükleme
with st.sidebar:
    st.header("📂 Dosya Yükle")
    uploaded_file = st.file_uploader("Bir PDF dosyası seçin", type="pdf")
    st.info("Motor: Llama 3.3 (Groq)")
    st.warning("Not: Site yenilendiğinde veriler sıfırlanır.")

# Veritabanı Hazırlama Fonksiyonu (Bulut İçin Özel)
@st.cache_resource
def pdf_islee(file):
    # Geçici bir klasör oluşturup dosyayı oraya kaydediyoruz
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    # PDF'i Oku ve Parçala
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Veritabanına Göm (Hafızada tutuyoruz, klasöre yazmıyoruz)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents=splits, embedding=embedding_model)
    
    # Geçici dosyayı temizle
    os.remove(tmp_path)
    return db

# --- ANA AKIŞ ---

if uploaded_file is None:
    # Dosya yoksa uyarı göster
    st.info("👈 Lütfen sol menüden bir PDF dosyası yükleyin.")
    st.image("https://cdn-icons-png.flaticon.com/512/337/337946.png", width=100) # Ok işareti

else:
    # Dosya varsa işle
    with st.spinner("PDF analiz ediliyor... (Bu işlem sadece bir kez yapılır)"):
        try:
            db = pdf_islee(uploaded_file)
            st.success("✅ PDF yüklendi! Sorunuzu sorabilirsiniz.")
            
            # Soru Kutusu
            soru = st.text_input("Sorunuzu yazın:", placeholder="Örn: Bu projenin amacı ne?")
            
            if st.button("Gönder 🚀") and soru:
                
                # Yapay Zeka Ayarı
                llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
                
                with st.spinner("Cevap hazırlanıyor..."):
                    # Benzerlik Araması
                    sonuclar = db.similarity_search(soru, k=4)
                    context = "\n\n".join([doc.page_content for doc in sonuclar])
                    
                    # Cevap Üretme
                    prompt = f"""
                    Aşağıdaki DOKÜMAN BİLGİSİ'ne göre SORU'yu Türkçe cevapla.
                    Bilgi yoksa "Dokümanda bulamadım" de.
                    
                    DOKÜMAN BİLGİSİ:
                    {context}
                    
                    SORU:
                    {soru}
                    """
                    cevap = llm.invoke(prompt)
                    
                    st.write("### 🤖 Cevap:")
                    st.write(cevap.content)
                    
                    with st.expander("Kaynaklar"):
                         for i, b in enumerate(sonuclar):
                            st.caption(f"**Parça {i+1}:** {b.page_content[:200]}...")

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")