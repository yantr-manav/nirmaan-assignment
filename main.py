import streamlit as st
import spacy
import requests
import nltk
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
import json
from datetime import datetime

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Nirmaan AI Evaluator", 
    layout="wide", 
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Pro" look
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4F46E5;
    }
    .stProgress > div > div > div > div {
        background-color: #4F46E5;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE CLASS (OOP Structure) ---
class RubricEngine:
    """
    Encapsulates the grading logic, keeping the UI separate from the Intelligence.
    """
    def __init__(self):
        self._load_resources()
        
    @st.cache_resource
    def _load_resources(_self):
        # NLP Core
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            
        # Semantic Brain
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sentiment
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        
        return nlp, embedder, sia

    def analyze(self, transcript, duration):
        nlp, embedder, sia = self._load_resources()
        doc = nlp(transcript)
        
        # 1. Compute Metrics
        metrics = {
            "word_count": len(doc),
            "wpm": (len(doc) / duration) * 60,
            "unique_words": len(set([t.text.lower() for t in doc if t.is_alpha])),
            "sentiment": sia.polarity_scores(transcript)['pos'],
            "fillers": [t.text for t in doc if t.text.lower() in ["um", "uh", "like", "actually", "basically"]],
            "entities": [(e.text, e.label_) for e in doc.ents]
        }
        
        # 2. Score & Generate Feedback
        results = {
            "scores": {},
            "feedback": {},
            "details": {}
        }
        
        # A. Content (Semantic Search)
        results['scores']['content'], results['details']['content'] = self._evaluate_content(doc, embedder)
        
        # B. Flow (Structure)
        results['scores']['flow'], results['details']['flow'] = self._evaluate_flow(transcript)
        
        # C. Speech Rate
        results['scores']['rate'], results['feedback']['rate'] = self._evaluate_rate(metrics['wpm'])
        
        # D. Grammar (API)
        results['scores']['grammar'], results['details']['grammar'] = self._evaluate_grammar_api(transcript)
        
        # E. Clarity (Fillers)
        filler_rate = (len(metrics['fillers']) / metrics['word_count']) * 100 if metrics['word_count'] > 0 else 0
        results['scores']['clarity'] = 15 if filler_rate < 3 else (10 if filler_rate < 7 else 5)
        results['details']['fillers'] = metrics['fillers']
        
        # F. Engagement
        results['scores']['engagement'] = 15 if metrics['sentiment'] > 0.2 else 8
        
        # Total
        results['total_score'] = sum(results['scores'].values())
        results['metrics'] = metrics
        
        return results

    def _evaluate_content(self, doc, embedder):
        score = 0
        details = []
        sentences = [sent.text for sent in doc.sents]
        
        # NER Check
        if any(e.label_ == "PERSON" for e in doc.ents):
            score += 5; details.append("✅ Name introduced")
        else: details.append("⚠️ Name missing")
            
        # Vector Similarity Check
        topics = {
            "Background/Family": ("I live with my family parents mother father.", 10),
            "Hobbies/Interests": ("I like playing reading hobbies sports music.", 10),
            "Ambition/Goals": ("I want to become aim goal future dream.", 5),
            "Schooling": ("I study in class school college grade.", 10)
        }
        
        for topic, (desc, weight) in topics.items():
            topic_emb = embedder.encode(desc, convert_to_tensor=True)
            max_sim = 0
            for sent in sentences:
                sent_emb = embedder.encode(sent, convert_to_tensor=True)
                sim = util.cos_sim(topic_emb, sent_emb).item()
                if sim > max_sim: max_sim = sim
            
            if max_sim > 0.25:
                score += weight
                details.append(f"✅ {topic} Covered")
            else:
                details.append(f"❌ {topic} Not clearly mentioned")
                
        return min(40, score), details

    def _evaluate_flow(self, text):
        text = text.lower()
        has_start = any(x in text[:50] for x in ['hello', 'hi', 'good', 'myself'])
        has_end = any(x in text[-50:] for x in ['thank', 'that\'s all', 'listening'])
        
        score = 0
        feedback = []
        if has_start: score += 2.5; feedback.append("Strong Opening")
        else: feedback.append("Abrupt Opening")
        
        if has_end: score += 2.5; feedback.append("Polite Closing")
        else: feedback.append("Abrupt Ending")
        
        return score, feedback

    def _evaluate_rate(self, wpm):
        if 110 <= wpm <= 150: return 10, "Perfect Pacing"
        elif 90 <= wpm <= 170: return 6, "Slightly Fast/Slow"
        else: return 2, "Needs pacing adjustment"

    def _evaluate_grammar_api(self, text):
        try:
            resp = requests.post("https://api.languagetool.org/v2/check", 
                               data={'text': text, 'language': 'en-US'}, timeout=3)
            if resp.ok:
                errors = resp.json().get('matches', [])
                score = max(0, 10 - len(errors))
                return score, errors
        except:
            pass
        return 10, [] # Fail open if API down

# --- VISUALIZATION HELPERS ---
def create_radar_chart(scores):
    labels = list(scores.keys())
    stats = list(scores.values())
    
    # Normalize to typical max values for visual balance
    max_vals = {'content': 40, 'flow': 5, 'rate': 10, 'grammar': 10, 'clarity': 15, 'engagement': 15}
    norm_stats = [min(100, (v / max_vals.get(k, 10)) * 100) for k, v in scores.items()]
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    norm_stats += norm_stats[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, norm_stats, color='#4F46E5', alpha=0.25)
    ax.plot(angles, norm_stats, color='#4F46E5', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.title() for l in labels], size=8)
    ax.set_yticklabels([])
    return fig

def create_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

# --- MAIN UI ---
def main():
    engine = RubricEngine()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712038.png", width=50)
        st.title("Nirmaan Coach")
        st.markdown("### Settings")
        duration = st.slider("Target Duration (sec)", 30, 120, 52)
        st.markdown("---")
        st.info("Built with Python, SpaCy & Transformer models.")

    # Main Content
    st.title("🎙️ AI Communication Assessment")
    st.markdown("Input your introduction transcript below to get an instant evaluation against the **Nirmaan Rubric**.")

    input_text = st.text_area("Transcript", height=200, placeholder="Hello everyone, my name is...")
    
    if st.button("Analyze Performance", type="primary"):
        if not input_text:
            st.warning("Please enter text first.")
            return
            
        with st.spinner("Computing Psychometrics & Semantics..."):
            results = engine.analyze(input_text, duration)
            
        # Top Level Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Score", f"{int(results['total_score'])}/100")
        col2.metric("WPM", f"{int(results['metrics']['wpm'])}")
        col3.metric("Vocabulary Size", results['metrics']['unique_words'])
        col4.metric("Fillers Found", len(results['metrics']['fillers']))
        
        st.markdown("---")
        
        # Detailed Analysis
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("📊 Skill Breakdown")
            st.pyplot(create_radar_chart(results['scores']))
            
        with c2:
            st.subheader("☁️ Topic Cloud")
            st.pyplot(create_wordcloud(input_text))

        # Actionable Feedback
        st.subheader("📝 Detailed Feedback Report")
        
        with st.expander("Content & Structure Analysis", expanded=True):
            st.markdown("**Semantic Coverage:**")
            for item in results['details']['content']:
                st.write(item)
            st.markdown("**Structure Flow:** " + ", ".join(results['details']['flow']))

        with st.expander("Grammar & Clarity Debugger"):
            if results['details']['grammar']:
                st.error(f"Found {len(results['details']['grammar'])} grammar issues.")
                for err in results['details']['grammar']:
                    st.write(f"- {err['message']} (Context: *{err['context']['text']}*)")
            else:
                st.success("No grammar errors detected!")
                
            if results['details']['fillers']:
                st.warning(f"Filler words used: {', '.join(set(results['details']['fillers']))}")

        # Export
        report_json = json.dumps(results, default=str, indent=2)
        st.download_button("Download Full Assessment Report", report_json, file_name="assessment.json")

if __name__ == "__main__":
    main()