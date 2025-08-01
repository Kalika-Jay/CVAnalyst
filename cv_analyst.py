import pdfplumber

def extract_text_from_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_similarity(resume_text, jd_text):
    if not resume_text or not jd_text:
        return 0.0

    documents = [resume_text, jd_text]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return 0.0


def find_missing_keywords(resume_text, jd_text, top_n=20):
    """Find important missing keywords using TF-IDF"""
    if not resume_text or not jd_text:
        return []

    try:
        # Get important terms from job description
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(1, 2))
        jd_tfidf = vectorizer.fit_transform([jd_text])

        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = jd_tfidf.toarray()[0]

        # Get top keywords by TF-IDF score
        keyword_scores = list(zip(feature_names, tfidf_scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)

        # Check which keywords are missing from resume
        resume_lower = resume_text.lower()
        missing_keywords = []

        for keyword, score in keyword_scores:
            if score > 0 and keyword not in resume_lower:
                missing_keywords.append(keyword)
                if len(missing_keywords) >= top_n:
                    break

        return missing_keywords
    except Exception as e:
        st.error(f"Error finding missing keywords: {str(e)}")
        return []
import streamlit as st

st.set_page_config(page_title="CVAnalyst", page_icon="📄", layout="wide")

st.title("📄 CVAnalyst")
st.markdown("Upload your resume and job description to get similarity analysis and optimization suggestions!")

st.markdown("""
    <style>
    .red-button > button {
        background-color: #e74c3c;
        color: white;
    }

    .green-button > button {
        background-color: #2ecc71;
        color: white;
    }

    .blue-button > button {
        background-color: #3498db;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Resume")
    resume_file = st.file_uploader("Choose a PDF file", type="pdf", key="resume")

with col2:
    st.subheader("📋 Job Description")
    jd_input = st.text_area("Paste the job description here", height=200, key="jd")
    st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

if resume_file and jd_input:
    with st.spinner("Analyzing your resume..."):
        # Extract and process text
        resume_text = extract_text_from_pdf(resume_file)

        if resume_text:
            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(jd_input)

            # Calculate similarity
            similarity = get_similarity(resume_clean, jd_clean)

            # Find missing keywords
            missing_keywords = find_missing_keywords(resume_clean, jd_clean)

            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")

            # Similarity score with color coding
            col1, col2, col3 = st.columns(3)

            with col1:
                if similarity >= 0.7:
                    st.success(f"🎯 **Excellent Match**")
                elif similarity >= 0.5:
                    st.warning(f"🤔 **Good Match**")
                else:
                    st.error(f"🔍 **Needs Improvement**")

            with col2:
                st.metric("Similarity Score", f"{similarity:.1%}")

            with col3:
                st.metric("Missing Keywords", f"{len(missing_keywords)}+")

            # Missing keywords section
            if missing_keywords:
                st.subheader("🔑 Missing Keywords")
                st.markdown("Consider incorporating these keywords from the job description:")

                # Display keywords in a nice format
                keywords_text = " • ".join(missing_keywords[:15])  # Show top 15
                st.info(keywords_text)

                if len(missing_keywords) > 15:
                    with st.expander("Show all missing keywords"):
                        all_keywords = " • ".join(missing_keywords)
                        st.write(all_keywords)

            # Recommendations
            st.subheader("💡 Recommendations")

            recommendations = []
            if similarity < 0.3:
                recommendations.append("🔄 Consider restructuring your resume to better match the job requirements")
            if similarity < 0.5:
                recommendations.append("📝 Add more relevant keywords and skills mentioned in the job description")
            if len(missing_keywords) > 20:
                recommendations.append("🎯 Focus on the most important missing keywords for better ATS compatibility")

            recommendations.extend([
                "📈 Quantify your achievements with specific numbers and metrics",
                "🎪 Use action verbs to describe your experiences",
                "📋 Tailor your resume summary to match the job requirements"
            ])

            for rec in recommendations:
                st.write(f"• {rec}")

            # Display extracted resume text (optional)
            with st.expander("📄 View extracted resume text"):
                st.text_area("Resume content:", resume_text, height=200, disabled=True)

else:
    st.info("👆 Please upload your resume (PDF) and paste the job description to get started!")

# Footer
st.markdown("---")
st.markdown(
    "💡 **Tip**: For best results, ensure your PDF is text-based (not scanned images) and the job description is complete.")
