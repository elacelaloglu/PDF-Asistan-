import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- AYARLAR ---
# GROQ ANAHTARINI BURAYA YAPIŞTIR (gsk_... ile başlar)
GROQ_API_KEY = "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"

# Sayfa Ayarları
st.set_page_config(page_title="Süper Hızlı Asistan", layout="wide")
st.title("⚡ Süper Hızlı Doküman Asistanı")

# Yan Menü
with st.sidebar:
    st.success("Motor: Llama 3.3 (Groq)")
    st.info("Dünyanın en yeni ve hızlı açık kaynak modeli.")

# 1. Veritabanını Yükle
@st.cache_resource
def veritabani_yukle():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
        return db
    except Exception as e:
        return None

db = veritabani_yukle()

if not db:
    st.error("Veritabanı bulunamadı.")
    st.stop()

# 2. Yapay Zekayı Başlat (GÜNCEL MODEL)
try:
    llm = ChatGroq(
        temperature=0, 
        # !!! İŞTE DEĞİŞİKLİK BURADA !!!
        # Eski model yerine en yeni ve en güçlü modeli yazdık.
        model_name="llama-3.3-70b-versatile", 
        api_key=GROQ_API_KEY
    )
except Exception as e:
    st.error(f"API Anahtarı hatası: {e}")
    st.stop()

# 3. Arayüz
soru = st.text_input("Sorunuzu yazın:", placeholder="Örn: Proje yürütücüsü kim?")

if st.button("Soruyu Gönder 🚀"):
    if not soru:
        st.warning("Lütfen bir soru yazın.")
    else:
        with st.spinner("Dokümanlar taranıyor..."):
            sonuclar = db.similarity_search(soru, k=4)
            bilgi_havuzu = ""
            for belge in sonuclar:
                bilgi_havuzu += belge.page_content + "\n\n"
        
        with st.spinner("Llama 3.3 düşünüyor..."):
            try:
                prompt = f"""
                Aşağıdaki BİLGİ'ye göre SORU'yu Türkçe cevapla.
                Bilgi içinde cevap yoksa "Dokümanlarda bulamadım" de.
                
                BİLGİ:
                {bilgi_havuzu}
                
                SORU:
                {soru}
                """
                
                cevap = llm.invoke(prompt)
                
                st.success("✅ Cevap:")
                st.write(cevap.content)
                
                with st.expander("Kaynak Paragraflar"):
                    for i, b in enumerate(sonuclar):
                        st.markdown(f"**Parça {i+1}:**")
                        st.caption(b.page_content)
                        st.divider()

            except Exception as e:
                st.error(f"Hata oluştu: {e}")