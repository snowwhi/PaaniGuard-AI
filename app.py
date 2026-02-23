"""
PaaniGuard — Hugging Face Spaces Version (Gradio 6.x)
"""

import os
import zipfile
import pickle
import gradio as gr

# ── LIBRARIES ─────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    LIBRARIES_OK = True
except Exception as e:
    LIBRARIES_OK = False
    print(f"⚠️ Missing libraries: {e}")

# ── GEMINI ────────────────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_AVAILABLE = False

if GEMINI_KEY:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        GEMINI_AVAILABLE = True
        print("✅ Gemini API key found and library loaded")
    except Exception as e:
        print(f"⚠️ langchain_google_genai import failed: {e}")
else:
    print("⚠️ GOOGLE_API_KEY secret not set in Space settings")

# ── DATABASE PATHS ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR   = os.path.join(BASE_DIR, "simple_faiss_db")
ZIP_PATH = os.path.join(BASE_DIR, "simple_faiss_db.zip")

print(f"📁 BASE_DIR : {BASE_DIR}")
print(f"📁 DB_DIR   : {DB_DIR}  exists={os.path.exists(DB_DIR)}")
print(f"📦 ZIP_PATH : {ZIP_PATH} exists={os.path.exists(ZIP_PATH)}")

# Extract zip if DB folder is missing
if not os.path.exists(DB_DIR) and os.path.exists(ZIP_PATH):
    print("📂 Extracting database zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(BASE_DIR)
    print("✅ Extraction done")
    # Fix nested folder if zip was zipped with folder inside
    nested = os.path.join(DB_DIR, "simple_faiss_db")
    if os.path.exists(nested):
        import shutil
        tmp = DB_DIR + "_tmp"
        shutil.move(nested, tmp)
        shutil.rmtree(DB_DIR)
        shutil.move(tmp, DB_DIR)
        print("✅ Fixed nested folder structure")

# ── LOAD DATABASE ─────────────────────────────────────────────────
model     = None
index     = None
chunks    = []
metadata  = []
db_loaded = False

if os.path.exists(DB_DIR) and LIBRARIES_OK:
    try:
        print("⏳ Loading SentenceTransformer...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("⏳ Loading FAISS index...")
        index = faiss.read_index(os.path.join(DB_DIR, "index.faiss"))
        with open(os.path.join(DB_DIR, "chunks.pkl"),   "rb") as f:
            chunks   = pickle.load(f)
        with open(os.path.join(DB_DIR, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        db_loaded = True
        print(f"✅ Database loaded — {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ DB load error: {e}")
else:
    if not os.path.exists(DB_DIR):
        print("❌ DB folder missing — check zip is uploaded to the Space repo")
    if not LIBRARIES_OK:
        print("❌ Required libraries not installed")

# ── SEARCH ────────────────────────────────────────────────────────
def search_db(query, k=3):
    if not db_loaded:
        return []
    query_vec = model.encode([query]).astype('float32')
    _, indices = index.search(query_vec, k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append({
                'text':   chunks[idx],
                'source': metadata[idx].get('source', 'Unknown'),
                'page':   metadata[idx].get('page',   'Unknown'),
            })
    return results

# ── CHAT ──────────────────────────────────────────────────────────
def chat(message, history):
    if not message:
        return "", history
    if history is None:
        history = []

    context = ""
    if db_loaded:
        results = search_db(message)
        if results:
            context = "\n\n".join(
                f"📄 From {r['source']} (Page {r['page']}):\n{r['text'][:500]}"
                for r in results
            )

    if GEMINI_AVAILABLE and GEMINI_KEY:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GEMINI_KEY,
                temperature=0.3,
            )
            if context:
                prompt = (
                    f"Based on these documents, answer the question:\n\n"
                    f"{context}\n\nQuestion: {message}\n\n"
                    f"Answer concisely using only the information above."
                )
            else:
                prompt = f"You are a Pakistan water expert. Answer:\n\n{message}"

            reply = llm.invoke(prompt).content
        except Exception as e:
            reply = f"⚠️ Gemini error: {e}"
    else:
        msg = message.lower()
        if context:
            reply = f"📚 Found in database:\n\n{context[:600]}"
        elif "flood"       in msg: reply = "🌊 Floods in Punjab occur during monsoon (July–September) along Chenab and Ravi."
        elif "balochistan" in msg: reply = "💧 Balochistan faces severe water scarcity; groundwater drops ~3.5 m/year."
        elif "indus"       in msg: reply = "🏞️ Indus River is 3,180 km long with tributaries Jhelum, Chenab, and Ravi."
        elif "lahore" in msg or "groundwater" in msg:
            reply = "⛰️ Lahore groundwater drops 2–3 m/year; ~60% of tube wells affected."
        else:
            reply = "Ask me about floods, water scarcity, the Indus River, or groundwater in Pakistan."

    # Gradio 6.x messages format
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history

# ── UI ────────────────────────────────────────────────────────────
with gr.Blocks(title="PaaniGuard") as demo:
    gr.Markdown("# 🛡️ PaaniGuard\nPakistan Water Intelligence System")

    db_color  = "#238636" if db_loaded       else "#e31a1c"
    db_label  = f"✅ Database: {len(chunks)} docs" if db_loaded else "❌ Database Not Found"
    gem_color = "#238636" if GEMINI_AVAILABLE else "#e31a1c"
    gem_label = "✅ Gemini Ready"            if GEMINI_AVAILABLE else "❌ Gemini Not Configured"

    gr.HTML(f"""
    <div style="text-align:center; margin:10px 0;">
        <span style="background:{db_color};  color:#fff; padding:5px 14px; border-radius:20px; margin:4px; display:inline-block;">{db_label}</span>
        <span style="background:{gem_color}; color:#fff; padding:5px 14px; border-radius:20px; margin:4px; display:inline-block;">{gem_label}</span>
    </div>
    """)

    gr.HTML("""
    <div style="display:flex; gap:10px; margin:16px 0;">
        <div style="flex:1;background:#161b22;padding:14px;border-radius:8px;text-align:center;">
            <div style="font-size:22px;color:#4fc3f7;font-weight:bold;">46K</div><div style="color:#aaa;">Records</div>
        </div>
        <div style="flex:1;background:#161b22;padding:14px;border-radius:8px;text-align:center;">
            <div style="font-size:22px;color:#4fc3f7;font-weight:bold;">13.5K</div><div style="color:#aaa;">Canals</div>
        </div>
        <div style="flex:1;background:#161b22;padding:14px;border-radius:8px;text-align:center;">
            <div style="font-size:22px;color:#4fc3f7;font-weight:bold;">7.8K</div><div style="color:#aaa;">Rivers</div>
        </div>
        <div style="flex:1;background:#161b22;padding:14px;border-radius:8px;text-align:center;">
            <div style="font-size:22px;color:#4fc3f7;font-weight:bold;">810</div><div style="color:#aaa;">Dams</div>
        </div>
    </div>
    """)

    # Gradio 6.x: type="messages" is required
    chatbot = gr.Chatbot(height=420, type="messages")

    with gr.Row():
        msg  = gr.Textbox(placeholder="Ask about Pakistan water...", scale=4, container=False)
        send = gr.Button("Send", scale=1, variant="primary")

    with gr.Row():
        gr.Button("🌊 Flood Punjab").click(
            lambda h: chat("Flood risk in Punjab", h), [chatbot], [msg, chatbot])
        gr.Button("💧 Water Balochistan").click(
            lambda h: chat("Water scarcity in Balochistan", h), [chatbot], [msg, chatbot])
        gr.Button("🏞️ Indus River").click(
            lambda h: chat("Indus River system", h), [chatbot], [msg, chatbot])
        gr.Button("⛰️ Groundwater Lahore").click(
            lambda h: chat("Groundwater in Lahore", h), [chatbot], [msg, chatbot])

    send.click(chat, [msg, chatbot], [msg, chatbot])
    msg.submit(chat, [msg, chatbot], [msg, chatbot])

demo.launch()
