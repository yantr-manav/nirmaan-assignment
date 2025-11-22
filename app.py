import streamlit as st
import spacy
import requests
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util

# --- 1. SETUP & CACHING ---
st.set_page_config(page_title="Nirmaan AI Grader Pro", layout="wide", page_icon="🎓")

@st.cache_resource
def load_models():
    # Load SpaCy for NER (Named Entity Recognition)
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    # Load SBERT for Semantic Similarity (The "Brain" of the app)
    # This helps us understand if the student talked about "hobbies" without using the exact word
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load Sentiment Analyzer
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    return nlp, embedder, sia

# Load models once and cache them
with st.spinner("Loading AI Models... (This may take a minute initially)"):
    nlp, embedder, sia = load_models()

# --- 2. CORE LOGIC FUNCTIONS ---

def check_grammar_api(text):
    """
    Uses the Public LanguageTool API directly via HTTP requests.
    This bypasses the need for Java installation locally.
    """
    url = "https://api.languagetool.org/v2/check"
    data = {
        'text': text,
        'language': 'en-US'
    }
    try:
        response = requests.post(url, data=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            return result.get('matches', [])
        else:
            st.warning(f"Grammar API Status: {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Grammar Check skipped (Network Error): {e}")
        return []

def check_semantic_topic(text_sentences, topic_description, threshold=0.25):
    """
    Checks if a specific topic (like 'hobbies') matches ANY sentence 
    in the student's speech using Vector Embeddings.
    """
    topic_embedding = embedder.encode(topic_description, convert_to_tensor=True)
    
    max_score = 0
    best_sentence = ""
    
    for sentence in text_sentences:
        sent_embedding = embedder.encode(sentence, convert_to_tensor=True)
        score = util.cos_sim(topic_embedding, sent_embedding).item()
        if score > max_score:
            max_score = score
            best_sentence = sentence
            
    return max_score >= threshold, max_score, best_sentence

def analyze_flow(doc):
    """
    Checks if the student follows: Salutation -> Body -> Closing
    """
    text_lower = doc.text.lower()
    
    salutations = ["hello", "hi", "good morning", "good afternoon", "myself", "hey"]
    closings = ["thank you", "thanks", "that's all", "listening", "nice meeting"]
    
    salutation_idx = -1
    closing_idx = -1
    
    # Find first occurrence of salutation
    for word in salutations:
        idx = text_lower.find(word)
        if idx != -1:
            salutation_idx = idx
            break
            
    # Find last occurrence of closing
    for word in closings:
        idx = text_lower.rfind(word)
        if idx != -1:
            closing_idx = idx
            break
    
    score = 0
    feedback = []
    
    # Rule: Salutation should be in the first 25% of the text
    if salutation_idx != -1 and salutation_idx < len(text_lower) * 0.25:
        score += 2.5
        feedback.append("✅ Proper Start")
    else:
        feedback.append("⚠️ Missing/Late Salutation")
        
    # Rule: Closing should be in the last 20% of the text
    if closing_idx != -1 and closing_idx > len(text_lower) * 0.75:
        score += 2.5
        feedback.append("✅ Proper Closing")
    else:
        feedback.append("⚠️ Missing/Early Closing")
        
    return score, ", ".join(feedback)

def advanced_content_analysis(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    score = 0
    details = []
    
    # A. Named Entity Check (NER)
    has_person = any(ent.label_ == "PERSON" for ent in doc.ents)
    has_loc = any(ent.label_ == "GPE" for ent in doc.ents)
    
    if has_person: 
        score += 4; details.append("✅ Name Detected (NER)")
    else:
        details.append("❌ Name Not Detected")

    if has_loc: 
        score += 2; details.append("✅ Location Detected")
    
    # B. Semantic Topic Checks
    # We compare student text against "Ideal Definitions"
    topics = {
        "Family": ("I live with my parents mother father siblings family.", 4),
        "Hobbies": ("I like playing cricket reading dancing drawing sports.", 4),
        "Ambition": ("I want to become a doctor engineer scientist leader.", 2),
        "School/Class": ("I study in class school grade college.", 4)
    }
    
    for topic, (desc, weight) in topics.items():
        present, similarity, match_sent = check_semantic_topic(sentences, desc)
        if present:
            score += weight
            details.append(f"✅ {topic} (Sim: {similarity:.2f})")
        else:
            details.append(f"❌ {topic} Missing")
            
    return min(score, 40), details

def plot_radar_chart(scores_dict):
    """Generates a Radar Chart for the UI"""
    labels = list(scores_dict.keys())
    stats = list(scores_dict.values())
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    stats += stats[:1]
    angles += angles[:1]
    labels += labels[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='#4F46E5', alpha=0.25)
    ax.plot(angles, stats, color='#4F46E5', linewidth=2)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=10)
    
    return fig

# --- 3. UI LAYOUT ---

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Customize the grading logic here.")
    min_duration = st.slider("Target Duration (sec)", 30, 90, 52)
    st.info("This dashboard demonstrates the 'Product Thinking' approach required by Nirmaan.")

st.title("🎓 Advanced Communication AI")
st.markdown("### Automated Assessment System | Nirmaan Case Study")
st.markdown("This tool uses **NER (Named Entity Recognition)**, **Vector Embeddings**, and **Grammar APIs** to grade student introductions.")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📝 Transcript Input")
    transcript_text = st.text_area("Paste the student's speech transcript:", height=250, placeholder="Hello everyone, myself...")
    
    c1, c2 = st.columns(2)
    with c1:
        duration_val = st.number_input("Audio Duration (seconds)", value=52)
    with c2:
        st.write("") # Spacer
        st.write("") # Spacer
        analyze_btn = st.button("🚀 Run Deep Analysis", type="primary", use_container_width=True)

if analyze_btn and transcript_text:
    with st.spinner("Running Semantic Analysis & NLP Pipeline..."):
        
        # 1. PRE-PROCESSING
        doc = nlp(transcript_text)
        word_count = len(doc)
        
        # 2. CONTENT & SEMANTICS (Max 40)
        content_score, content_details = advanced_content_analysis(transcript_text)
        
        # 3. FLOW (Max 5)
        flow_score, flow_feedback = analyze_flow(doc)
        
        # 4. SPEECH RATE (Max 10)
        wpm = (word_count / duration_val) * 60
        if 110 <= wpm <= 150: rate_score = 10
        elif 90 <= wpm <= 170: rate_score = 6
        else: rate_score = 2
        
        # 5. GRAMMAR (Max 10) - USING API NOW
        errors = check_grammar_api(transcript_text)
        grammar_score = max(0, 10 - len(errors))
        
        # 6. VOCABULARY (Max 10)
        unique_words = set([token.text.lower() for token in doc if token.is_alpha])
        ttr = len(unique_words) / word_count if word_count > 0 else 0
        vocab_score = 10 if ttr > 0.6 else (6 if ttr > 0.4 else 4)
        
        # 7. FILLERS (Max 15)
        filler_list = ["um", "uh", "like", "you know", "basically", "actually"]
        fillers_found = [token.text for token in doc if token.text.lower() in filler_list]
        filler_rate = (len(fillers_found) / word_count) * 100 if word_count > 0 else 0
        
        if filler_rate < 3: clarity_score = 15
        elif filler_rate < 8: clarity_score = 10
        else: clarity_score = 5
        
        # 8. SENTIMENT (Max 10)
        polarity = sia.polarity_scores(transcript_text)
        pos_score = polarity['pos']
        sentiment_score = 10 if pos_score > 0.15 else 5
        
        # TOTAL
        total_score = content_score + flow_score + rate_score + grammar_score + vocab_score + clarity_score + sentiment_score
        total_score = min(100, total_score)

    # --- RESULTS COLUMN ---
    with col2:
        st.subheader("📊 Analysis Results")
        
        st.metric(label="Final Weighted Score", value=f"{int(total_score)} / 100")
        
        # Radar Chart Data
        chart_data = {
            "Content": (content_score / 40) * 100,
            "Flow": (flow_score / 5) * 100,
            "Rate": (rate_score / 10) * 100,
            "Grammar": (grammar_score / 10) * 100,
            "Clarity": (clarity_score / 15) * 100
        }
        st.pyplot(plot_radar_chart(chart_data))

    # --- DETAILED BREAKDOWN ---
    st.markdown("---")
    st.subheader("🔎 Detailed Criterion Breakdown")
    
    tab1, tab2, tab3 = st.tabs(["Content & Flow", "Language & Clarity", "Debug Data"])
    
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Content Analysis (NLP)")
            for item in content_details:
                st.markdown(f"- {item}")
        with col_b:
            st.markdown("#### Structural Flow")
            st.info(f"Status: {flow_feedback}")

    with tab2:
        st.markdown(f"**Grammar Issues:** {len(errors)} found")
        if errors:
            with st.expander("View Grammar Errors"):
                for err in errors:
                    msg = err.get('message', 'Error')
                    ctx = err.get('context', {}).get('text', 'Unknown context')
                    st.write(f"- {msg} (Context: '{ctx}')")
        
        st.markdown(f"**Vocabulary (TTR):** {ttr:.2f} (Richness score)")
        st.markdown(f"**Filler Words:** {len(fillers_found)} found")

    with tab3:
        st.json({
            "word_count": word_count,
            "wpm": wpm,
            "sentiment_pos": polarity['pos'],
            "entities_detected": [(ent.text, ent.label_) for ent in doc.ents]
        })