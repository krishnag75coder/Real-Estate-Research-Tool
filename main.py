import streamlit as st
from rag import process_urls, generate_answer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Real Estate Research Tool",
    page_icon="🏠",
    layout="wide"
)

# ---------------- DARK THEME CSS ----------------
st.markdown("""
<style>
body, .stApp { background-color: #0b0f17; }
h1 { text-align:center; color:#00eaff; text-shadow:0 0 12px #00eaff55; }
h2, h3, h4 { color:#ffffff !important; }
label { color:#cfd8dc !important; }
.stTextInput input { background-color:#111827 !important; color:white !important; border-radius:12px; border:1px solid #00eaff55; padding:12px; }
.stButton button { width:100%; border-radius:12px; background:linear-gradient(90deg,#00eaff,#007cf0); color:white; font-weight:600; height:3em; border:none; box-shadow:0 0 15px #00eaff33; }
.stButton button:hover { transform:scale(1.03); box-shadow:0 0 25px #00eaffaa; transition:.2s; }
section[data-testid="stSidebar"] { background:#0f172a; }
p { color:#e0e0e0; }
::-webkit-scrollbar { width:8px; }
::-webkit-scrollbar-thumb { background:#00eaff55; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🏠 Real Estate Research Tool")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔗 Enter URLs")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")
placeholder = st.empty()
process_url_button = st.sidebar.button("⚡ Process URLs")

# ---------------- PROCESS URLS ----------------
if process_url_button:
    urls = [u for u in (url1, url2, url3) if u.strip() != ""]
    if not urls:
        placeholder.warning("⚠️ Enter at least one URL")
    else:
        for status in process_urls(urls):
            placeholder.info("⏳ " + status)

# ---------------- QUERY ----------------
st.markdown("### Ask a Question")
query = st.text_input("Type your question...")

if query:
    try:
        with st.spinner("🤖 Generating Answer..."):
            answer, sources = generate_answer(query)

        st.markdown("## 📊 Answer")
        st.write(answer)

        if sources:
            st.markdown("### 📚 Sources")
            import re
            source_list = re.split(r'[\n, ]+', sources.strip())
            for source in source_list:
                if source:
                    st.markdown(f"🔗 {source}")

    except RuntimeError:
        st.error("⚠️ Please process URLs first.")