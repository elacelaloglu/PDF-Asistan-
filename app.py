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

# Sayfa Ayarları
st.set_page_config(page_title="Pro Asistan", layout="wide")
st.title("🧠 Profesyonel Doküman Asistanı")
st.markdown("Birden fazla PDF yükleyin, sohbet edin ve detaylı analizler alın.")

# --- SESSION STATE (HAFIZA) ---
# Sohbet geçmişini burada tutacağız
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

# --- YAN MENÜ (DOSYA YÜKLEME) ---
with st.sidebar:
    st.header("📂 Doküman Yönetimi")
    # accept_multiple_files=True ile çoklu seçim açıldı
    uploaded_files = st.file_uploader("PDF Dosyalarını Seçin", type="pdf", accept_multiple_files=True)
    
    process_btn = st.button("Dokümanları Analiz Et ⚡")
    
    st.divider()
    st.info("Model: Llama 3.3 (Groq)")
    if st.button("Sohbeti Temizle 🗑️"):
        st.session_state.messages = []
        st.rerun()

# --- FONKSİYONLAR ---
def veritabani_olustur(files):
    documents = []
    
    # İlerleme çubuğu
    progress_bar = st.progress(0)
    
    for i, file in enumerate(files):
        # Her dosyayı geçici olarak kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        # Oku
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        documents.extend(docs) # Listeye ekle
        
        # Dosyayı sil
        os.remove(tmp_path)
        
        # İlerlemeyi güncelle
        progress_bar.progress((i + 1) / len(files))

    # Parçala
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Veritabanına Göm
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents=splits, embedding=embedding_model)
    
    progress_bar.empty() # Çubuğu temizle
    return db

# --- İŞLEM AKIŞI ---

# 1. Dokümanları İşle (Butona basılınca)
if process_btn and uploaded_files:
    with st.spinner("Dokümanlar birleştiriliyor ve yapay zeka tarafından okunuyor..."):
        try:
            st.session_state.db = veritabani_olustur(uploaded_files)
            st.success(f"✅ Toplam {len(uploaded_files)} dosya başarıyla işlendi!")
            st.session_state.messages = [] # Yeni dosya gelince sohbeti sıfırla
        except Exception as e:
            st.error(f"Hata: {e}")

# 2. Sohbet Geçmişini Ekrana Yaz
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Kullanıcıdan Yeni Soru Al (Chat Input)
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    
    # Veritabanı kontrolü
    if st.session_state.db is None:
        st.warning("Lütfen önce sol taraftan PDF yükleyip 'Analiz Et' butonuna basın.")
    else:
        # Kullanıcı mesajını ekrana ve hafızaya ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Cevap Üretme
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyor..."):
                try:
                    # RAG - Arama
                    db = st.session_state.db
                    sonuclar = db.similarity_search(prompt, k=5)
                    context = "\n\n".join([doc.page_content for doc in sonuclar])
                    
                    # Sohbet Geçmişini Metne Çevir (Hafıza)
                    gecmis_sohbet = ""
                    for msg in st.session_state.messages[-6:]: # Son 6 mesajı hatırla (Hız için)
                        gecmis_sohbet += f"{msg['role']}: {msg['content']}\n"

                    # Gelişmiş Prompt (Detaylı Cevap İçin)
                    system_prompt = f"""
                    Sen uzman bir kurumsal asistansın. Görevin verilen dokümanlara dayanarak detaylı, profesyonel ve açıklayıcı cevaplar vermektir.
                    
                    KURALLAR:
                    1. Cevapların doyurucu ve uzun olsun. Maddeler halinde açıklama yapmayı tercih et.
                    2. Sohbet geçmişini dikkate al. Kullanıcı "O kim?" derse, geçmişten kimden bahsettiğini anla.
                    3. Bilgiyi sadece aşağıdaki DOKÜMAN içeriğinden al. Uydurma yapma.
                    
                    SOHBET GEÇMİŞİ:
                    {gecmis_sohbet}
                    
                    DOKÜMAN BİLGİSİ:
                    {context}
                    
                    KULLANICI SORUSU:
                    {prompt}
                    """
                    
                    # Groq'a Gönder
                    llm = ChatGroq(temperature=0.3, model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
                    response = llm.invoke(system_prompt)
                    cevap = response.content
                    
                    # Cevabı Yaz
                    st.markdown(cevap)
                    
                    # Hafızaya Kaydet
                    st.session_state.messages.append({"role": "assistant", "content": cevap})
                    
                    # Kaynakları göster (Expander içinde)
                    with st.expander("Referans Kaynaklar"):
                        for i, doc in enumerate(sonuclar):
                            st.caption(f"**Kaynak {i+1}:** {doc.page_content[:200]}...")

                except Exception as e:
                    st.error(f"Hata oluştu: {e}")