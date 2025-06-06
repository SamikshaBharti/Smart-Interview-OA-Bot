import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="Smart Interview Bot",
    layout="wide",
    page_icon="🤖"
)

# FINAL DARK THEME CSS (ChatGPT style)
custom_css = """
<style>
body, .reportview-container, .main, .block-container {
    background-color: #000000 !important;
    color: #e0e0e0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.css-18e3th9, .css-1dp5vir, .css-1v0mbdj, .css-12oz5g7 {
    background-color: #000000 !important;
}

h1 {
    font-size: 3.2rem !important;
    color: #00c6ff;
    font-weight: 900;
    text-shadow: 1px 1px 5px #00c6ffaa;
}

h2, h3 {
    color: #00aaffcc;
    font-weight: 700;
}

textarea {
    background-color: #000000 !important;
    border: 2px solid #00c6ff !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-size: 1.2rem !important;
    padding: 15px !important;
}

.stButton > button {
    background: #00c6ff;
    color: #000000;
    font-weight: 700;
    font-size: 1.25rem;
    padding: 12px 0;
    border-radius: 12px;
    width: 100%;
    transition: background 0.3s ease;
}
.stButton > button:hover {
    background: #0284c7;
    color: #fff;
}

.stInfo, .stWarning, .stSuccess {
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 0 15px #00c6ff66;
    background: rgba(0, 198, 255, 0.15) !important;
    color: #cceeff !important;
}

.question-card {
    background: linear-gradient(135deg, #021124, #0b2a4a);
    border-radius: 16px;
    padding: 15px 20px;
    margin-bottom: 12px;
    transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.3s ease;
    cursor: pointer;
    box-shadow: 0 0 10px #00c6ff55;
    color: #ffffff;
    font-size: 1.1rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-shadow: 0 0 3px #00000099;
}
.question-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 25px #00c6ffcc;
    background: linear-gradient(135deg, #0077b6, #00c6ff);
    color: #ffffff;
}

.scroll-container {
    max-height: 300px;
    overflow-y: auto;
    padding-right: 10px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Load data
df = pd.read_csv("C:\\Users\\HP\\OneDrive\\Desktop\\Smart_Interview_and_OA_Bot\\interview_data.csv")

df.dropna(subset=['Question'], inplace=True)
df['Question'] = df['Question'].str.lower().str.strip()
df['Topic'] = df['Topic'].fillna('misc')
df['Difficulty'] = df['Difficulty'].fillna('medium')
if 'Company' not in df.columns:
    df['Company'] = 'general'

train_company_model = df['Company'].nunique() > 1

X = df['Question']
y_topic = df['Topic']
y_difficulty = df['Difficulty']
y_company = df['Company']

# Models
topic_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression(max_iter=200))
])
topic_model.fit(X, y_topic)

difficulty_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', RandomForestClassifier())
])
difficulty_model.fit(X, y_difficulty)

if train_company_model:
    company_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression(max_iter=200))
    ])
    company_model.fit(X, y_company)

vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(df['Question'])

# App Header
st.title("Smart Interview & OA Bot")

st.markdown(
    """
    <p style="font-size:1.3rem; max-width: 900px; line-height:1.6; color:#a9d6e5;">
    Enter any interview question or topic below. This bot will intelligently:
    <ul>
        <li>Predict the <strong>Topic</strong> and <strong>Difficulty</strong></li>
        <li>Guess the <strong>Company</strong> if data permits</li>
        <li>Show you the best matching and similar questions with a sleek interface</li>
    </ul>
    </p>
    """,
    unsafe_allow_html=True
)

# Input area
user_input = st.text_area("🧠 Your Question or Topic:", height=150)

if user_input:
    if st.button("🔍 Analyze Question"):
        user_input_clean = user_input.lower().strip()

        pred_topic = topic_model.predict([user_input_clean])[0]
        pred_diff = difficulty_model.predict([user_input_clean])[0]
        pred_company = company_model.predict([user_input_clean])[0] if train_company_model else None

        q_vec = vectorizer.transform([user_input_clean])
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()

        threshold = 0.3
        best_idx = sims.argmax()
        best_score = sims[best_idx]

        col1, col2 = st.columns([3, 7])

        with col1:
            if best_score < threshold:
                st.warning("⚠️ No exact question found in the database.")
                st.markdown(
                    f"""
                    <div class="question-card" title="Closest Match (Similarity: {best_score:.2f})">
                        <strong>Closest Question Found:</strong><br>{df.iloc[best_idx]['Question']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
                    <div class="question-card">
                        <strong>Topic:</strong> Unknown<br>
                        <strong>Difficulty:</strong> Unknown<br>
                        <strong>Company:</strong> Unknown<br>
                        <small>This question has some similarity with the one shown above.</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.success(f"📌 Topic: **{pred_topic}**")
                st.info(f"🎯 Difficulty: **{pred_diff}**")
                if pred_company:
                    st.warning(f"🏢 Company: **{pred_company}**")
                else:
                    st.warning("🏢 Company prediction unavailable.")

        with col2:
            if best_score >= threshold:
                st.subheader("🔥 Best Match Question")
                st.markdown(
                    f"""
                    <div class="question-card">{df.iloc[best_idx]['Question']}</div>
                    """,
                    unsafe_allow_html=True
                )

                st.subheader("📚 Similar Questions")
                st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
                top_k = sims.argsort()[-5:][::-1]
                for i in top_k:
                    st.markdown(
                        f'<div class="question-card">{df.iloc[i]["Question"]}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("💡 Please enter your question or topic to start.")
