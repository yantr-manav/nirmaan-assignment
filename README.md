# 🎙️ Nirmaan AI Communication Grader

> **An intelligent assessment engine that evaluates student introductions using NLP, Semantic Analysis, and Psycholinguistic metrics.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![SpaCy](https://img.shields.io/badge/NLP-SpaCy-green)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

## 🚀 Overview

This project was engineered as a case study for the **Nirmaan Organization AI Internship**. It automates the grading of student self-introductions based on a specific educational rubric.

Unlike traditional keyword-counting scripts, this tool employs **Vector Embeddings (SBERT)** and **Named Entity Recognition (NER)** to understand the *semantics* and *context* of a student's speech, providing a human-like assessment of their communication skills.

### 🌟 Key Features
* **Semantic Scoring:** Uses `sentence-transformers` to credit students for covering topics (e.g., "Family") even if they use synonyms, rather than exact keyword matching.
* **Entity Recognition:** Uses **SpaCy NER** to validate that specific entities (Names, Locations) are actually present.
* **Cloud-Based Grammar:** Bypasses heavy local Java dependencies by integrating directly with the **LanguageTool API** for real-time grammar checking.
* **Visual Analytics:** Generates professional **Radar Charts** and **Topic Word Clouds** for intuitive feedback.
* **Report Export:** Allows users to download a full JSON assessment report for record-keeping.

---

## 🛠️ Tech Stack & Architecture

The application follows a modular **Object-Oriented** design, separating the grading logic (`RubricEngine`) from the User Interface.

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Frontend** | `Streamlit` | Rapid prototyping, responsive UI, and native support for data visualization. |
| **NLP Core** | `SpaCy` (`en_core_web_sm`) | Efficient tokenization and robust Named Entity Recognition (NER). |
| **Semantics** | `all-MiniLM-L6-v2` | A high-performance, lightweight Transformer model ideal for CPU-based inference. |
| **Grammar** | `LanguageTool API` | Accurate rule-based grammar checking without complex local JVM setups. |
| **Visualization** | `Matplotlib` & `WordCloud` | For generating standard academic charts and visual summaries. |

---

## ⚙️ Installation & Setup

Follow these steps to deploy the application locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/yantr-manav/nirmaan-assignment.git](https://github.com/yantr-manav/nirmaan-assignment.git)

cd nirmaan-assignment
```

### 2. Create environment and Install Dependencies

``` 
python -m venv myenv

myenv\Scripts\activate
(Windows)

pip install -r requirements.txt
```

### 3. Download NLP Models
The app requires the SpaCy English model to function.

```
python -m spacy download en_core_web_sm
```
### 4. Run the Applicationn

```
streamlit run main.py
```
The application will launch automatically at ` http://localhost:8501 `

## 🧠 How the Scoring Engine Works


The `RubricEngine` class calculates scores based on the provided Nirmaan Excel rubric:

### 1. Content (40%):

* **NER:** Checks for `PERSON` and `GPE` (Location) entities to ensure factual introductions.

* **Vectors:** Compares the student's sentences against "Anchor Sentences" (e.g., "I live with my family") using Cosine Similarity. If similarity > 0.25, the topic is marked as covered.

### 2. Structure (5%): 
Analyzes the first 20% and last 20% of the transcript for specific salutations and closing markers (Flow detection).

### 3. Speech Rate (10%): 
Calculates Words Per Minute (WPM) based on input duration vs. word count.

### 4. Grammar (10%): 
Deductive scoring based on the error density returned by the LanguageTool API.

### 5. Engagement (15%): 
Uses VADER Sentiment Analysis to detect positive polarity and enthusiasm in the text.

---
### 📂 Project Structure
```
nirmaan-assignment/
├── main.py              # Main Application Entry Point & UI
├── requirements.txt    # Python Dependencies
├── README.md           # Documentation
└── assets/             # Images and demo resources
```

## 🔮 Future Improvements

Given more time, the following features would be priortized:
### 1. Audio Ingestion :
Integrate `OpenAI Whisper` to allow direct audio file uploads instead of text transcripts
### 2. LLM Feedback: 
Use a generative model ( Llama 3 or GPT-4o) to write personalized textual advice rather than just displaying metrics.
### 3. Database Integration :
Store student scores in SQLite/PostgreSQL to track progress over time.



---
Submitted by Saivamshi Jilla for the AI Internship Case Study