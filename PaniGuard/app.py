"""
🛡️ PaaniGuard — Pakistan Water Intelligence System
Features:
  1. AI Chat         — RAG + Gemini, grounded in 46,146 OSM waterway records
  2. Waterway Search — instant keyword search across the full database
  3. Report Writer   — generates a structured water crisis report on any topic
"""

# ── IMPORTS ───────────────────────────────────────────────────────
import os, re, zipfile, warnings, sqlite3
warnings.filterwarnings("ignore")

import gradio as gr
import plotly.graph_objects as go
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ── CONFIG ────────────────────────────────────────────────────────
CHROMA_DIR = "./paaniguard_index"
CHROMA_ZIP = "./paaniguard_index.zip"
DB_PATH    = "./paaniguard_index/chroma.sqlite3"
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "")

# ── STARTUP ───────────────────────────────────────────────────────
print("🚀 Booting PaaniGuard...")

if not os.path.exists(CHROMA_DIR):
    if os.path.exists(CHROMA_ZIP):
        print("📦 Extracting ChromaDB...")
        with zipfile.ZipFile(CHROMA_ZIP, "r") as z:
            z.extractall("./")
        print("✅ ChromaDB extracted.")

print("🧠 Loading embedding model...")
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("🗄️  Connecting to vector store...")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedder,
    collection_name="langchain",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

print("🤖 Initialising Gemini...")
llm = ChatGoogleGenerativeAI(
   model="gemini-2.5-flash",
    google_api_key=GEMINI_KEY,
    temperature=0.3,
    max_output_tokens=2000,
) if GEMINI_KEY else None

print("✅ All systems GO.\n")

# ── PROMPTS ───────────────────────────────────────────────────────
CHAT_PROMPT = """You are PaaniGuard, an expert AI for Pakistan water infrastructure and crisis management.

DATABASE (46,146 OSM waterway records):
13,543 canals | 7,809 rivers | 12,143 streams | 6,357 ditches | 3,668 drains | 1,374 wadis | 810 dams

EXPERTISE: Indus, Jhelum, Chenab, Ravi, Sutlej, Kabul rivers; Punjab/Sindh/KPK canal networks;
WAPDA, IRSA, PCRWR; flood zones, water scarcity, climate impact, Indus Waters Treaty.

RULES:
- Cite actual waterway names from [CONTEXT]
- 150-200 words, sharp and factual
- For flood/crisis: name affected regions, give action steps
- Never invent waterway names
- Reply in user's language (Urdu or English)

"""

REPORT_PROMPT = """You are PaaniGuard Report Writer. Write a professional water crisis report.

Use exactly these ## sections:
## Executive Summary
## Current Situation
## Key Waterways & Infrastructure Affected
## Root Causes
## Immediate Recommendations
## Long-Term Solutions

Rules: cite waterway names from [CONTEXT], be factual and actionable, 400-500 words total.

"""

# ── HELPERS ───────────────────────────────────────────────────────
def retrieve_context(query: str) -> str:
    try:
        docs = retriever.get_relevant_documents(query)
        if docs:
            return "\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
        return "No records found."
    except Exception as e:
        return f"Retrieval error: {e}"

def ask_gemini(system_prompt: str, user_text: str) -> str:
    if not llm:
        return "⚠️ GOOGLE_API_KEY not set. Add it in Space Settings → Secrets."
    try:
        # Combine system + user into single HumanMessage (Gemini requirement)
        full_prompt = f"{system_prompt}\n\n{user_text}"
        resp = llm.invoke(full_prompt)
        return resp.content
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# ── STATS CHART ───────────────────────────────────────────────────
def build_stats_chart():
    categories = ["Canal", "Stream", "River", "Ditch", "Drain", "Wadi", "Dam", "Weir"]
    counts     = [13543, 12143, 7809, 6357, 3668, 1374, 810, 292]
    colors     = ["#2ca25f","#74c476","#1a6faf","#fdae6b",
                  "#fd8d3c","#bcbddc","#e31a1c","#9ecae1"]
    fig = go.Figure(go.Bar(
        x=categories, y=counts, marker_color=colors,
        text=[f"{c:,}" for c in counts],
        textposition="outside",
        textfont=dict(color="white", size=10),
    ))
    fig.update_layout(
        title=dict(text="Pakistan Waterway Database — 46,146 Records",
                   font=dict(color="white", size=13)),
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=250, font=dict(color="white"),
        margin=dict(l=10, r=10, t=44, b=10),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#21262d"),
    )
    return fig

# ── FEATURE 1: CHAT ───────────────────────────────────────────────
def chat_fn(message: str, history: list):
    if not message.strip():
        return "", history
    context = retrieve_context(message)
    prompt  = f"[CONTEXT]\n{context}\n\n[QUESTION]\n{message}"
    reply   = ask_gemini(CHAT_PROMPT, prompt)
    return "", history + [(message, reply)]

# ── FEATURE 2: SEARCH ─────────────────────────────────────────────
def search_waterways(query: str):
    if not query.strip():
        return "Enter a search term above.", None

    query_clean = query.strip().lower()
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("""
            SELECT DISTINCT string_value
            FROM embedding_metadata
            WHERE key = 'chroma:document'
              AND LOWER(string_value) LIKE ?
            LIMIT 200
        """, (f"%{query_clean}%",))
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
    except Exception as e:
        return f"Database error: {e}", None

    if not rows:
        return f"No waterways found matching **'{query}'**.", None

    parsed = []
    for r in rows:
        name_match = re.search(r"Waterway: (.+?)\s*\(", r)
        type_match = re.search(r"Type: (\w+)", r)
        urdu_match = re.search(r"\((.+?)\)", r)
        name  = name_match.group(1).strip() if name_match else "Unknown"
        wtype = type_match.group(1).capitalize() if type_match else "Unknown"
        urdu  = urdu_match.group(1).strip() if urdu_match else "—"
        if urdu == "نام دستیاب نہیں":
            urdu = "—"
        parsed.append((name, wtype, urdu))

    seen, unique = set(), []
    for p in parsed:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    lines = [
        f"### 🔍 {len(unique)} results for '{query}'\n",
        "| # | Waterway Name | Type | Urdu Name |",
        "|---|--------------|------|-----------|",
    ]
    for i, (name, wtype, urdu) in enumerate(unique[:50], 1):
        lines.append(f"| {i} | {name} | {wtype} | {urdu} |")
    if len(unique) > 50:
        lines.append(f"\n*... and {len(unique)-50} more results*")

    type_counts = {}
    for _, wtype, _ in unique:
        type_counts[wtype] = type_counts.get(wtype, 0) + 1
    type_counts = dict(sorted(type_counts.items(), key=lambda x: -x[1]))

    colors_map = {
        "River":"#1a6faf","Canal":"#2ca25f","Stream":"#74c476",
        "Drain":"#fd8d3c","Ditch":"#fdae6b","Wadi":"#bcbddc",
        "Dam":"#e31a1c","Weir":"#9ecae1",
    }
    fig = go.Figure(go.Bar(
        x=list(type_counts.keys()),
        y=list(type_counts.values()),
        marker_color=[colors_map.get(t, "#888") for t in type_counts],
        text=list(type_counts.values()),
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        title=dict(text=f"Type Breakdown — '{query}'", font=dict(color="white", size=12)),
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=240, font=dict(color="white"),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#21262d"),
    )
    return "\n".join(lines), fig

# ── FEATURE 3: REPORT ─────────────────────────────────────────────
def generate_report(topic: str) -> str:
    if not topic.strip():
        return "Please enter a topic."
    context = retrieve_context(topic)
    prompt  = f"[CONTEXT]\n{context}\n\n[REPORT TOPIC]\n{topic}"
    return ask_gemini(REPORT_PROMPT, prompt)

# ── CONSTANTS ─────────────────────────────────────────────────────
QUICK_QUERIES = [
    "Rivers flooding in Punjab monsoon?",
    "Canals in Sindh province",
    "Water scarcity in Balochistan",
    "Indus River system",
    "Bhakkar district waterways",
    "Tarbela and Mangla dams",
]
SEARCH_TERMS  = ["river","canal","nala","wadi","Indus","Chenab","Ravi","Jhelum"]
REPORT_TOPICS = [
    "Flood risk in Punjab during monsoon",
    "Water scarcity crisis in Balochistan",
    "Canal deterioration in Sindh",
    "Groundwater depletion in Lahore",
    "Climate change impact on Indus River",
]

# ── CSS — MOBILE FIRST ────────────────────────────────────────────
CSS = """
/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── Header ── */
.hdr {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
    border: 1px solid #2a4a6b;
}
.hdr h1 {
    color: #4fc3f7; font-size: clamp(1.3rem, 4vw, 2rem);
    margin: 0 0 3px; font-weight: 800;
}
.hdr p { color: #90caf9; margin: 0; font-size: clamp(0.72rem, 2vw, 0.88rem); }
.prose, .md, .markdown, textarea, .output-markdown p, .output-markdown li { color: #e6edf3 !important; }

/* ── Stat boxes ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
    gap: 8px; margin-bottom: 12px;
}
.sbox {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 10px 6px; text-align: center;
}
.sbox .n { font-size: clamp(1rem, 3vw, 1.4rem); font-weight: 700; color: #4fc3f7; }
.sbox .l { font-size: clamp(0.62rem, 1.5vw, 0.72rem); color: #8b949e; margin-top: 2px; }

/* ── Buttons ── */
.qb {
    background: #161b22 !important; border: 1px solid #2a4a6b !important;
    color: #90caf9 !important; font-size: clamp(0.7rem, 1.8vw, 0.8rem) !important;
    border-radius: 6px !important; padding: 6px 8px !important;
    white-space: normal !important; text-align: left !important;
    line-height: 1.3 !important; min-height: 36px !important;
}
.qb:hover { background: #1f2937 !important; border-color: #4fc3f7 !important; }
.send { background: #1f6feb !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; }
.gen  { background: #238636 !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; }
.srch { background: #8957e5 !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; }

/* ── Chatbot ── */
.chatbot-wrap { border-radius: 10px !important; }

/* ── Tabs ── */
.tab-nav button { font-size: clamp(0.78rem, 2vw, 0.9rem) !important; }

/* ── Responsive layout ── */
@media (max-width: 640px) {
    .gradio-container { padding: 8px !important; }
    .hdr { padding: 12px 14px; }
    .main-row { flex-direction: column !important; }
    .side-col { width: 100% !important; }
    .quick-grid {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 6px !important;
    }
}

/* ── Input ── */
input, textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus { border-color: #4fc3f7 !important; outline: none !important; }

footer { display: none !important; }
"""

# ── UI ────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="PaaniGuard 🛡️") as demo:

    # Header
    gr.HTML("""
    <div class="hdr">
        <h1>🛡️ PaaniGuard</h1>
        <p>Pakistan Water Intelligence &nbsp;·&nbsp; 46,146 OSM Records &nbsp;·&nbsp; RAG + Gemini AI</p>
    </div>""")

    # Stats
    gr.HTML("""
    <div class="stats-row">
        <div class="sbox"><div class="n">46K</div><div class="l">Records</div></div>
        <div class="sbox"><div class="n">13.5K</div><div class="l">Canals</div></div>
        <div class="sbox"><div class="n">7.8K</div><div class="l">Rivers</div></div>
        <div class="sbox"><div class="n">810</div><div class="l">Dams</div></div>
        <div class="sbox"><div class="n">1.4K</div><div class="l">Wadis</div></div>
    </div>""")

    # Stats chart
    stats_chart = gr.Plot(show_label=False)

    gr.HTML("<div style='height:8px'></div>")

    # ── TABS ──
    with gr.Tabs():

        # TAB 1 — CHAT
        with gr.Tab("💬 AI Chat"):
            chatbot = gr.Chatbot(
                height=380,
                show_label=False,
                placeholder="Ask about Pakistan's rivers, canals, floods, water scarcity...",
            )
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask anything about Pakistan water...",
                    show_label=False, scale=5, container=False,
                )
                chat_send = gr.Button("Send 🚀", scale=1, elem_classes=["send"], min_width=80)

            gr.HTML("<p style='color:#4fc3f7;font-size:0.82rem;margin:10px 0 6px'>⚡ Quick Questions</p>")
            # Mobile-friendly 2-column grid
            with gr.Row():
                qb1 = gr.Button(QUICK_QUERIES[0], elem_classes=["qb"])
                qb2 = gr.Button(QUICK_QUERIES[1], elem_classes=["qb"])
            with gr.Row():
                qb3 = gr.Button(QUICK_QUERIES[2], elem_classes=["qb"])
                qb4 = gr.Button(QUICK_QUERIES[3], elem_classes=["qb"])
            with gr.Row():
                qb5 = gr.Button(QUICK_QUERIES[4], elem_classes=["qb"])
                qb6 = gr.Button(QUICK_QUERIES[5], elem_classes=["qb"])

            chat_send.click(chat_fn, [chat_input, chatbot], [chat_input, chatbot])
            chat_input.submit(chat_fn, [chat_input, chatbot], [chat_input, chatbot])
            for btn, q in zip([qb1,qb2,qb3,qb4,qb5,qb6], QUICK_QUERIES):
                btn.click(lambda h, query=q: chat_fn(query, h),
                          inputs=[chatbot], outputs=[chat_input, chatbot])

        # TAB 2 — SEARCH
        with gr.Tab("🔍 Search"):
            gr.HTML("<p style='color:#8b949e;font-size:0.85rem;margin:0 0 10px'>"
                    "Search 46,146 waterway records instantly.</p>")
            with gr.Row():
                search_input = gr.Textbox(
                    placeholder="e.g. Indus, canal, nala, Ravi...",
                    show_label=False, scale=5, container=False,
                )
                search_btn = gr.Button("Search 🔎", scale=1, elem_classes=["srch"], min_width=90)

            gr.HTML("<p style='color:#4fc3f7;font-size:0.8rem;margin:8px 0 5px'>Quick searches:</p>")
            with gr.Row():
                sb1 = gr.Button("river",  elem_classes=["qb"])
                sb2 = gr.Button("canal",  elem_classes=["qb"])
                sb3 = gr.Button("nala",   elem_classes=["qb"])
                sb4 = gr.Button("wadi",   elem_classes=["qb"])
            with gr.Row():
                sb5 = gr.Button("Indus",  elem_classes=["qb"])
                sb6 = gr.Button("Chenab", elem_classes=["qb"])
                sb7 = gr.Button("Ravi",   elem_classes=["qb"])
                sb8 = gr.Button("Jhelum", elem_classes=["qb"])

            search_results = gr.Markdown("*Enter a search term to explore the database.*")
            search_chart   = gr.Plot(show_label=False)

            search_btn.click(search_waterways, [search_input], [search_results, search_chart])
            search_input.submit(search_waterways, [search_input], [search_results, search_chart])
            for btn, term in zip([sb1,sb2,sb3,sb4,sb5,sb6,sb7,sb8], SEARCH_TERMS):
                btn.click(lambda t=term: search_waterways(t),
                          outputs=[search_results, search_chart])

        # TAB 3 — REPORT
        with gr.Tab("📄 Report"):
            gr.HTML("<p style='color:#8b949e;font-size:0.85rem;margin:0 0 10px'>"
                    "Generate structured water crisis reports grounded in real data.</p>")
            with gr.Row():
                report_input = gr.Textbox(
                    placeholder="e.g. Flood risk in Punjab during monsoon...",
                    show_label=False, scale=5, container=False,
                )
                report_btn = gr.Button("Generate 📝", scale=1, elem_classes=["gen"], min_width=100)

            gr.HTML("<p style='color:#4fc3f7;font-size:0.8rem;margin:8px 0 5px'>Example topics:</p>")
            with gr.Row():
                rt1 = gr.Button(REPORT_TOPICS[0], elem_classes=["qb"])
                rt2 = gr.Button(REPORT_TOPICS[1], elem_classes=["qb"])
            with gr.Row():
                rt3 = gr.Button(REPORT_TOPICS[2], elem_classes=["qb"])
                rt4 = gr.Button(REPORT_TOPICS[3], elem_classes=["qb"])
            with gr.Row():
                rt5 = gr.Button(REPORT_TOPICS[4], elem_classes=["qb"])

            report_output = gr.Markdown(
                value="*Enter a topic and click Generate.*",
                height=450,
            )

            report_btn.click(generate_report, [report_input], [report_output])
            report_input.submit(generate_report, [report_input], [report_output])
            for btn, topic in zip([rt1,rt2,rt3,rt4,rt5], REPORT_TOPICS):
                btn.click(lambda t=topic: generate_report(t), outputs=[report_output])

    # Footer
    gr.HTML("""
    <div style='text-align:center;color:#484f58;font-size:0.72rem;
                margin-top:14px;padding:6px;border-top:1px solid #21262d'>
        PaaniGuard · OpenStreetMap ODbL · Feb 2026 · Pakistan Water Crisis Hackathon
    </div>""")

    demo.load(build_stats_chart, outputs=[stats_chart])

# ── LAUNCH ────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue(max_size=5, default_concurrency_limit=2)
    demo.launch()
