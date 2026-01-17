import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import pandas as pd
from deepseek_ocr import DeepSeekOCR  # Installeras via repo

@st.cache_resource
def load_model():
    return DeepSeekOCR.from_pretrained("deepseek-ai/DeepSeek-OCR")

st.title("DeepSeek OCR: Årsredovisning → CSV")
uploaded_file = st.file_uploader("Ladda upp årsredovisning (PDF)", type="pdf")

if uploaded_file:
    model = load_model()
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_md = ""
    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(2, 2))  # 144 DPI
        img = Image.open(pix.tobytes("png"))
        result = model.infer(img, prompt="Convert to detailed markdown with tables preserved. Swedish annual report.")
        full_md += result["text"] + "\n\n--- Page " + str(i+1) + " ---\n\n"

    # Extrahera specifika värden (anpassa nycklar)
    prompt_extract = """
    Extrahera från markdown: Omsättning, Resultat efter finansnetto, Årsresultat, 
    Totala tillgångar, Eget kapital. Returnera som JSON: {"Omsättning": "värde", "Resultat": "värde", ...}
    """
    extract_result = model.infer(full_md[:10000], prompt=prompt_extract)  # Chunk om stort
    data = eval(extract_result["text"])  # Parsa JSON (säker i prod med json.loads)

    df = pd.DataFrame([data])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Ladda ner CSV", csv, "arsredovisning.csv", "text/csv")
    st.dataframe(df)
    st.markdown("Raw Markdown: " + full_md[:2000])
