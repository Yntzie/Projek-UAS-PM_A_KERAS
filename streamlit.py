import streamlit as st
import torch
from unsloth import FastLanguageModel

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Chatbot UAJY - Dinar",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Asisten Kampus UAJY")
st.caption("Dibuat dengan Llama 3 & Sahabat-AI oleh Dinar")

# --- 2. LOAD MODEL (Hanya sekali di awal) ---
@st.cache_resource
def load_model():
    model_name = "Sahabat-AI/llama3-8b-cpt-sahabatai-v1-instruct"
    # Load model dasar + Adapter yang baru saja dilatih
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "model_uajy_final", # Load dari folder hasil training
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) # Aktifkan mode inferensi (lebih cepat)
    return model, tokenizer

# Tampilkan loading spinner saat memuat model
with st.spinner('Sedang menyiapkan otak AI... (Tunggu sebentar)'):
    model, tokenizer = load_model()

# --- 3. LOGIKA CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan history chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. INPUT USER & GENERATE JAWABAN ---
if prompt := st.chat_input("Tanya seputar UAJY..."):
    # Simpan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Format Prompt sesuai Sahabat-AI
    alpaca_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Anda adalah asisten AI Universitas Atma Jaya Yogyakarta (UAJY). Jawablah dengan sopan dan akurat.<|eot_id|><|start_header_id|>user<|end_header_id|>

    {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    inputs = tokenizer(
        [alpaca_prompt.format(prompt)], 
        return_tensors = "pt"
    ).to("cuda")

    # Generate jawaban
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Streaming output (biar terlihat mengetik)
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 256, 
            use_cache = True,
            pad_token_id = tokenizer.eos_token_id
        )
        
        # Decode hasil (menghilangkan prompt input)
        response_text = tokenizer.batch_decode(outputs)[0]
        # Kita potong bagian prompt agar hanya ambil jawabannya
        response_cleaned = response_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "")
        
        message_placeholder.markdown(response_cleaned)
    
    # Simpan jawaban bot
    st.session_state.messages.append({"role": "assistant", "content": response_cleaned})