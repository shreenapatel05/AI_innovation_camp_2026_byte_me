
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Integrated Health Analytics",
    page_icon="🏥",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stMetric {
    background-color: #1a1d24;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #2d3139;
}
.stMetric label { color: #00d9ff !important; }
h1 { color: #00d9ff; font-weight: 700; }
h2, h3 { color: #ffffff; }
.stTabs [data-baseweb="tab"] {
    background-color: #1a1d24;
    border-radius: 5px;
    padding: 10px 20px;
    color: #ffffff;
}
.stTabs [aria-selected="true"] {
    background-color: #00d9ff;
    color: #000000;
}
.emergency-high {
    background-color: #ff4444;
    color: white;
    padding: 15px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}
.emergency-medium {
    background-color: #ffaa00;
    color: white;
    padding: 15px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}
.emergency-low {
    background-color: #00ff88;
    color: black;
    padding: 15px;
    border-radius: 10px;
    font-weight: bold;
    text-align: center;
}
.india-card {
    background: linear-gradient(135deg, #1a1d24 0%, #2a2d34 100%);
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #ff9933;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_deaths_data():
    """Load cause of deaths dataset"""
    try:
        import kagglehub
        path = kagglehub.dataset_download("iamsouravbanerjee/cause-of-deaths-around-the-world")
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

        if csv_files:
            df = pd.read_csv(os.path.join(path, csv_files[0]))
            df.columns = df.columns.str.strip()
            df = df.fillna(0).drop_duplicates()

            if 'Year' in df.columns:
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df = df.dropna(subset=['Year'])
                df['Year'] = df['Year'].astype(int)
            return df
    except Exception as e:
        st.error(f"Error loading deaths data: {e}")
        return None

@st.cache_data
def load_symptoms_data():
    """Load patient symptoms dataset"""
    paths = [
        "patient_symptoms_textual.csv",
        "patient_symptoms_textual__1_.csv",
        "patient_symptoms_textual (1).csv"
    ]

    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.lower().str.strip()
            df = df.dropna(subset=["patient_concern", "diagnosis"])
            df["patient_concern"] = df["patient_concern"].str.lower().str.strip()
            return df
    return None

@st.cache_data
def load_indian_disease_data():
    """Load Indian disease symptom dataset"""
    paths = [
        "indian_disease_symptom_5000.csv",
        "/content/indian_disease_symptom_5000.csv"
    ]

    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            return df
    return None

@st.cache_resource
def train_symptom_model():
    """Train symptom prediction model"""
    df = load_symptoms_data()
    if df is None:
        return None, 0

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

    X = df["patient_concern"]
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_cause_columns(df):
    exclude = ['Country', 'Territory', 'Code', 'Year']
    return [col for col in df.columns
            if not any(p.lower() in col.lower() for p in exclude)]

def calculate_total_deaths(df, cause_cols):
    for col in cause_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['Total_Deaths'] = df[cause_cols].sum(axis=1)
    return df

def format_number(num):
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    return f"{num:.0f}"

def get_country_column(df):
    for col in df.columns:
        if 'country' in col.lower() and 'code' not in col.lower():
            return col
    return 'Country/Territory'

def detect_emergency(symptom):
    """Emergency level detection"""
    high = ["chest pain", "breathing difficulty", "heavy breathing", "cardiac"]
    medium = ["severe headache", "vomiting", "wheezing", "dizziness", "blurred vision"]

    sym_lower = symptom.lower()
    if any(k in sym_lower for k in high):
        return "HIGH"
    if any(k in sym_lower for k in medium):
        return "MEDIUM"
    return "LOW"

def get_remedies(disease):
    """Home remedies database"""
    remedies_db = {
        "migraine": ["💧 Drink 2L water daily", "📱 Reduce screen time", "😴 Sleep 8 hours", "🌑 Rest in dark room"],
        "flu": ["💊 Paracetamol (doctor-advised)", "🌡️ Warm water", "🛏️ Rest", "🍲 Light food"],
        "influenza": ["💊 Paracetamol (doctor-advised)", "🌡️ Warm water", "🛏️ Rest", "🍲 Light food"],
        "common cold": ["🌡️ Warm fluids", "🍯 Honey-ginger tea", "😴 Rest", "💨 Steam inhalation"],
        "common_cold": ["🌡️ Warm fluids", "🍯 Honey-ginger tea", "😴 Rest", "💨 Steam inhalation"],
        "uti": ["💧 Plenty of water", "🍋 Cranberry juice", "👨‍⚕️ Complete antibiotics"],
        "food poisoning": ["💧 ORS solution", "🍚 Light khichdi", "🚫 Avoid oily food", "🍌 BRAT diet"],
        "acidity": ["🥛 Curd rice", "🚫 Avoid spicy food", "🥗 Small frequent meals"],
        "gastritis": ["🥛 Curd rice", "🚫 Avoid spicy food", "🥗 Small frequent meals"],
        "throat infection": ["🌡️ Salt water gargle", "🍯 Honey", "🍋 Lemon tea", "🚫 Cold drinks"],
        "asthma": ["💊 Use inhaler", "🚫 Avoid triggers", "🧘 Breathing exercises"],
        "cardiac issue": ["🚨 URGENT MEDICAL ATTENTION", "💊 Prescribed medications", "🚫 Avoid stress"],
        "diabetes": ["🍎 Monitor blood sugar", "🏃 Regular exercise", "🥗 Balanced diet"],
        "arthritis": ["🏃 Gentle exercise", "🔥 Hot/cold therapy", "💊 Anti-inflammatory"],
        "malaria": ["💊 Complete antimalarial course", "💧 Stay hydrated", "😴 Rest well"],
        "pneumonia": ["💊 Complete antibiotics", "🛏️ Complete rest", "💧 Hydration", "🌡️ Monitor fever"],
        "bronchitis": ["💨 Steam inhalation", "💧 Warm fluids", "🛏️ Rest", "🚫 Avoid irritants"],
        "anemia": ["🥬 Iron-rich foods", "🍊 Vitamin C", "💊 Iron supplements", "🩺 Regular checkups"],
        "hypertension": ["🧂 Reduce salt intake", "🏃 Regular exercise", "😌 Stress management", "💊 Take medications"],
        "anxiety": ["🧘 Meditation", "🏃 Regular exercise", "😴 Adequate sleep", "🗣️ Talk therapy"],
        "depression": ["🗣️ Professional counseling", "🏃 Physical activity", "😴 Sleep routine", "👥 Social support"],
    }

    disease_low = disease.lower().replace(" ", "_")
    for key, val in remedies_db.items():
        if key in disease_low or disease_low in key:
            return val
    return ["💊 Consult doctor", "💧 Stay hydrated", "😴 Get rest"]

def get_doctor_specialist(disease):
    """Specialist recommendations"""
    doctors = {
        "migraine": "🧠 Neurologist",
        "cardiac issue": "❤️ Cardiologist (URGENT)",
        "asthma": "🫁 Pulmonologist",
        "food poisoning": "🔬 Gastroenterologist",
        "acidity": "🔬 Gastroenterologist",
        "gastritis": "🔬 Gastroenterologist",
        "uti": "🔬 Urologist / General Physician",
        "diabetes": "🩺 Endocrinologist",
        "arthritis": "🦴 Rheumatologist",
        "throat infection": "👂 ENT Specialist",
        "flu": "🩺 General Physician",
        "influenza": "🩺 General Physician",
        "common cold": "🩺 General Physician",
        "common_cold": "🩺 General Physician",
        "malaria": "🩺 General Physician",
        "pneumonia": "🫁 Pulmonologist / General Physician",
        "bronchitis": "🫁 Pulmonologist",
        "anemia": "🩺 Hematologist",
        "hypertension": "❤️ Cardiologist",
        "anxiety": "🧠 Psychiatrist / Psychologist",
        "depression": "🧠 Psychiatrist / Psychologist",
    }

    disease_low = disease.lower().replace(" ", "_")
    for key, val in doctors.items():
        if key in disease_low or disease_low in key:
            return val
    return "🩺 General Physician"

def get_diet_plan(disease):
    """Diet recommendations"""
    diet_db = {
        "acidity": {
            "recommended": ["🥛 Curd rice", "🥔 Boiled vegetables", "🍌 Bananas"],
            "avoid": ["🌶️ Spicy food", "☕ Coffee", "🍫 Chocolate"]
        },
        "gastritis": {
            "recommended": ["🥛 Curd rice", "🥔 Boiled vegetables", "🍌 Bananas"],
            "avoid": ["🌶️ Spicy food", "☕ Coffee", "🍫 Chocolate"]
        },
        "flu": {
            "recommended": ["🍲 Warm soup", "🍚 Khichdi", "🍊 Vitamin C fruits"],
            "avoid": ["🥶 Cold drinks", "🍦 Ice cream"]
        },
        "influenza": {
            "recommended": ["🍲 Warm soup", "🍚 Khichdi", "🍊 Vitamin C fruits"],
            "avoid": ["🥶 Cold drinks", "🍦 Ice cream"]
        },
        "diabetes": {
            "recommended": ["🥗 Green vegetables", "🌾 Whole grains", "🐟 Lean protein"],
            "avoid": ["🍰 Sweets", "🍞 White bread", "🧃 Sugary drinks"]
        },
        "cardiac issue": {
            "recommended": ["🥗 Leafy greens", "🐟 Fish", "🌾 Oats", "🥜 Nuts"],
            "avoid": ["🧂 High salt", "🍟 Fried food", "🥓 Red meat"]
        },
        "hypertension": {
            "recommended": ["🥗 Leafy greens", "🐟 Fish", "🌾 Oats", "🥜 Nuts"],
            "avoid": ["🧂 High salt", "🍟 Fried food", "🥓 Red meat"]
        },
        "uti": {
            "recommended": ["💧 Water", "🍋 Cranberry juice", "🥒 Cucumber"],
            "avoid": ["☕ Caffeine", "🍺 Alcohol", "🌶️ Spicy food"]
        },
        "anemia": {
            "recommended": ["🥬 Spinach", "🥩 Red meat", "🍊 Citrus fruits", "🌰 Dry fruits"],
            "avoid": ["☕ Tea with meals", "🥛 Excess calcium", "🍺 Alcohol"]
        },
    }

    disease_low = disease.lower().replace(" ", "_")
    for key, val in diet_db.items():
        if key in disease_low or disease_low in key:
            return val

    return {
        "recommended": ["🥗 Balanced diet", "💧 Plenty of water", "🍎 Fresh fruits"],
        "avoid": ["🍟 Junk food", "🌶️ Very spicy food"]
    }

# ============================================================================
# MAIN APP
# ============================================================================

def main():

    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 5px;'>
            🏥 Integrated Health Analytics Dashboard
        </h1>
        <p style='text-align: center; color: #888; font-size: 16px;'>
            Global Health Statistics • AI Symptom Checker • India Disease Analysis
        </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Load data
    with st.spinner("📥 Loading datasets and training AI model..."):
        deaths_df = load_deaths_data()
        symptoms_model, model_accuracy = train_symptom_model()
        indian_df = load_indian_disease_data()

    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/health-data.png", width=100)
    st.sidebar.title("📊 Dashboard")
    st.sidebar.markdown("---")

    if symptoms_model is not None:
        st.sidebar.success(f"🤖 AI Model Ready\nAccuracy: {model_accuracy*100:.1f}%")

    if indian_df is not None:
        st.sidebar.info(f"🇮🇳 India Dataset\n{len(indian_df):,} records")

    # Create tabs
    tabs = st.tabs([
        "🤖 AI Symptom Checker",
        "🇮🇳 India Analysis",
        "🌍 Global Overview",
        "🗺️ Country Analysis",
        "🦠 Disease Impact",
        "📈 Time Series"
    ])

    # ========================================================================
    # TAB 1: AI SYMPTOM CHECKER
    # ========================================================================

    with tabs[0]:
        st.header("🤖 AI Medical Symptom Checker")

        if symptoms_model is None:
            st.error("❌ Symptom prediction model not available. Please ensure patient_symptoms_textual.csv is uploaded.")
        else:
            st.markdown("""
            <div style='background-color: #1a1d24; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h4 style='color: #00d9ff;'>ℹ️ How to Use</h4>
                <p>Describe your symptoms in natural language. Our AI will predict possible diagnoses
                with recommendations. <strong>This is for informational purposes only</strong> and not a
                substitute for professional medical advice.</p>
            </div>
            """, unsafe_allow_html=True)

            # Model statistics
            symptoms_df = load_symptoms_data()


            st.markdown("---")

            # Symptom input
            st.subheader("💬 Describe Your Symptoms")

            col_left, col_right = st.columns([2, 1])

            with col_left:
                user_symptoms = st.text_area(
                    "Enter your symptoms here:",
                    height=120,
                    placeholder="e.g., I have severe headache with sensitivity to light and nausea...",
                    help="Be as detailed as possible for better predictions"
                )

                analyze_button = st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True)

            with col_right:
                st.markdown("**💡 Example Symptoms:**")
                examples = [
                    "Headache with light sensitivity",
                    "Chest pain and breathing difficulty",
                    "Burning while urinating",
                    "High fever with body ache",
                    "Stomach pain after eating"
                ]

                for ex in examples:
                    if st.button(ex, key=ex, use_container_width=True):
                        user_symptoms = ex
                        analyze_button = True

            # Analysis
            if analyze_button and user_symptoms:

                with st.spinner("🔄 Analyzing symptoms..."):

                    cleaned_symptoms = user_symptoms.lower().strip()

                    # Prediction
                    predicted_disease = symptoms_model.predict([cleaned_symptoms])[0]
                    proba = symptoms_model.predict_proba([cleaned_symptoms])[0]
                    confidence = max(proba) * 100
                    emergency_level = detect_emergency(cleaned_symptoms)

                    st.markdown("---")
                    st.markdown("### 📋 Analysis Results")

                    result_col1, result_col2 = st.columns([1, 1])

                    with result_col1:
                        st.markdown(f"""
                        <div style='background-color: #1a1d24; padding: 20px; border-radius: 10px; border-left: 5px solid #00d9ff;'>
                            <h3 style='color: #00d9ff; margin-top: 0;'>🩺 Predicted Diagnosis</h3>
                            <h2 style='color: white; margin: 10px 0;'>{predicted_disease}</h2>
                            <p style='color: #888;'>Confidence: <strong style='color: #00ff88;'>{confidence:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                    with result_col2:
                        emergency_class = f"emergency-{emergency_level.lower()}"
                        emergency_icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}[emergency_level]

                        st.markdown(f"""
                        <div class='{emergency_class}'>
                            <h3 style='margin-top: 0;'>{emergency_icon} Emergency Level</h3>
                            <h2 style='margin: 10px 0;'>{emergency_level}</h2>
                            <p style='margin-bottom: 0;'>
                                {'🚨 Seek immediate medical attention!' if emergency_level == 'HIGH'
                                 else '⚠️ Consult a doctor soon.' if emergency_level == 'MEDIUM'
                                 else '✅ Monitor symptoms, consult if worsens.'}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    # Recommendations
                    rec_tabs = st.tabs(["🏠 Home Remedies", "👨‍⚕️ Doctor", "🍽️ Diet", "📊 Similar Cases"])

                    with rec_tabs[0]:
                        st.subheader("🏠 Recommended Home Remedies")
                        remedies = get_remedies(predicted_disease)

                        for i, remedy in enumerate(remedies, 1):
                            st.markdown(f"{i}. {remedy}")

                        st.warning("⚠️ These are general recommendations. Always consult a healthcare professional.")

                    with rec_tabs[1]:
                        st.subheader("👨‍⚕️ Recommended Specialist")
                        specialist = get_doctor_specialist(predicted_disease)

                        st.markdown(f"""
                        <div style='background-color: #1a1d24; padding: 25px; border-radius: 10px; text-align: center;'>
                            <h2 style='color: #00d9ff; margin: 0;'>{specialist}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                        st.info("💡 Book an appointment with this specialist for proper diagnosis and treatment.")

                    with rec_tabs[2]:
                        st.subheader("🍽️ Diet Recommendations")
                        diet = get_diet_plan(predicted_disease)

                        diet_col1, diet_col2 = st.columns(2)

                        with diet_col1:
                            st.markdown("**✅ Recommended Foods:**")
                            for food in diet["recommended"]:
                                st.markdown(f"- {food}")

                        with diet_col2:
                            st.markdown("**❌ Foods to Avoid:**")
                            for food in diet["avoid"]:
                                st.markdown(f"- {food}")

                    with rec_tabs[3]:
                        st.subheader("📊 Similar Cases in Database")

                        similar_cases = symptoms_df[symptoms_df['diagnosis'] == predicted_disease].head(5)

                        st.markdown(f"**Found {len(similar_cases)} similar cases:**")

                        for idx, case in similar_cases.iterrows():
                            st.markdown(f"""
                            <div style='background-color: #1a1d24; padding: 15px; border-radius: 5px; margin-bottom: 10px;'>
                                <p style='color: #00d9ff; margin: 0;'><strong>Case {idx + 1}:</strong></p>
                                <p style='margin: 5px 0;'>{case['patient_concern']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Disclaimer
                    st.markdown("---")
                    st.error("""
                    **⚠️ MEDICAL DISCLAIMER:**
                    This AI tool is for informational and educational purposes only. It is NOT a substitute
                    for professional medical advice, diagnosis, or treatment. Always seek the advice of your
                    physician or other qualified health provider.
                    """)

            # Dataset visualization
            st.markdown("---")
            st.subheader("📊 Dataset Insights")

            vis_col1, vis_col2 = st.columns(2)

            with vis_col1:
                disease_counts = symptoms_df['diagnosis'].value_counts().head(10)

                fig = go.Figure(data=[
                    go.Bar(
                        x=disease_counts.values,
                        y=disease_counts.index,
                        orientation='h',
                        marker=dict(color=disease_counts.values, colorscale='Viridis', showscale=True),
                        text=disease_counts.values,
                        textposition='auto',
                    )
                ])

                fig.update_layout(
                    title="Top 10 Diagnoses in Dataset",
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#2d3139', title='Count'),
                    yaxis=dict(gridcolor='#2d3139', title=''),
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            with vis_col2:
                st.markdown("**Common Symptoms WordCloud**")

                text = " ".join(symptoms_df["patient_concern"].astype(str))
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color="#0e1117",
                    colormap='viridis'
                ).generate(text)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                fig.patch.set_facecolor('#0e1117')

                st.pyplot(fig)

    # ========================================================================
    # TAB 2: INDIA ANALYSIS (NEW)
    # ========================================================================

    with tabs[1]:
        st.header("🇮🇳 India Disease Analysis")

        if indian_df is None:
            st.error("❌ Indian disease dataset not available. Please upload indian_disease_symptom_5000.csv")
        else:
            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("📊 Total Records", f"{len(indian_df):,}")
            col2.metric("🦠 Diseases", indian_df['Disease'].nunique())
            col3.metric("🗺️ States/UTs", indian_df['State'].nunique())
            col4.metric("👥 Avg Age", f"{indian_df['Age'].mean():.1f} years")

            st.markdown("---")

            # State-wise disease heatmap
            st.subheader("🗺️ State-wise Disease Distribution Heatmap")

            state_disease = indian_df.groupby(['State', 'Disease']).size().reset_index(name='Cases')

            # Get top diseases
            top_diseases_india = indian_df['Disease'].value_counts().head(10).index
            state_disease_filtered = state_disease[state_disease['Disease'].isin(top_diseases_india)]

            # Create pivot table for heatmap
            heatmap_data = state_disease_filtered.pivot(index='State', columns='Disease', values='Cases').fillna(0)

            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Reds',
                text=heatmap_data.values.astype(int),
                texttemplate='%{text}',
                textfont={"size": 8},
                colorbar=dict(title="Cases")
            ))

            fig.update_layout(
                title="Disease Distribution Across Indian States",
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                height=700,
                xaxis=dict(tickangle=-45),
                yaxis=dict(autorange='reversed')
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # State selector
            col_select, col_info = st.columns([2, 1])

            with col_select:
                selected_state = st.selectbox(
                    "🔍 Select a State/UT for Detailed Analysis",
                    sorted(indian_df['State'].unique())
                )

            state_data = indian_df[indian_df['State'] == selected_state]

            with col_info:
                st.markdown(f"""
                <div class='india-card'>
                    <h4 style='color: #00d9ff; margin: 0;'>{selected_state}</h4>
                    <p style='margin: 5px 0;'>📊 Records: {len(state_data)}</p>
                    <p style='margin: 5px 0;'>🦠 Diseases: {state_data['Disease'].nunique()}</p>
                </div>
                """, unsafe_allow_html=True)

            # State analysis
            state_tabs = st.tabs([
                "🦠 Disease Distribution",
                "👥 Demographics",
                "🩺 Vital Signs",
                "📊 Symptom Analysis"
            ])

            with state_tabs[0]:
                st.subheader(f"Top Diseases in {selected_state}")

                disease_counts_state = state_data['Disease'].value_counts().head(10)

                fig = go.Figure(data=[
                    go.Bar(
                        x=disease_counts_state.values,
                        y=disease_counts_state.index,
                        orientation='h',
                        marker=dict(
                            color=disease_counts_state.values,
                            colorscale='Oranges',
                            showscale=True
                        ),
                        text=disease_counts_state.values,
                        textposition='auto'
                    )
                ])

                fig.update_layout(
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#2d3139', title='Cases'),
                    yaxis=dict(gridcolor='#2d3139'),
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            with state_tabs[1]:
                st.subheader("Demographic Analysis")

                demo_col1, demo_col2 = st.columns(2)

                with demo_col1:
                    # Age distribution
                    fig = go.Figure(data=[
                        go.Histogram(
                            x=state_data['Age'],
                            nbinsx=15,
                            marker=dict(color='#00d9ff', line=dict(color='white', width=1))
                        )
                    ])

                    fig.update_layout(
                        title="Age Distribution",
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        xaxis=dict(gridcolor='#2d3139', title='Age'),
                        yaxis=dict(gridcolor='#2d3139', title='Count'),
                        height=350
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with demo_col2:
                    # Gender distribution
                    gender_counts = state_data['Gender'].value_counts()

                    fig = go.Figure(data=[
                        go.Pie(
                            labels=gender_counts.index,
                            values=gender_counts.values,
                            hole=0.4,
                            marker=dict(colors=['#00d9ff', '#ff006e'])
                        )
                    ])

                    fig.update_layout(
                        title="Gender Distribution",
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        height=350
                    )

                    st.plotly_chart(fig, use_container_width=True)

            with state_tabs[2]:
                st.subheader("Vital Signs Analysis")

                vital_col1, vital_col2, vital_col3 = st.columns(3)

                with vital_col1:
                    st.metric("🩸 Avg BP", f"{state_data['BloodPressure'].mean():.0f} mmHg")

                    fig = go.Figure(data=[
                        go.Box(
                            y=state_data['BloodPressure'],
                            marker_color='#ff006e',
                            name='BP'
                        )
                    ])

                    fig.update_layout(
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        height=250,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with vital_col2:
                    st.metric("❤️ Avg HR", f"{state_data['HeartRate'].mean():.0f} bpm")

                    fig = go.Figure(data=[
                        go.Box(
                            y=state_data['HeartRate'],
                            marker_color='#8338ec',
                            name='HR'
                        )
                    ])

                    fig.update_layout(
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        height=250,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with vital_col3:
                    st.metric("🌡️ Avg Temp", f"{state_data['Temperature'].mean():.1f}°F")

                    fig = go.Figure(data=[
                        go.Box(
                            y=state_data['Temperature'],
                            marker_color='#ffbe0b',
                            name='Temp'
                        )
                    ])

                    fig.update_layout(
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        height=250,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

            with state_tabs[3]:
                st.subheader("Common Symptoms Analysis")

                # Get all symptom columns
                symptom_cols = ['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5']

                # Count all symptoms
                all_symptoms = []
                for col in symptom_cols:
                    all_symptoms.extend(state_data[col].dropna().tolist())

                symptom_counts = pd.Series(all_symptoms).value_counts().head(15)

                fig = go.Figure(data=[
                    go.Bar(
                        x=symptom_counts.index,
                        y=symptom_counts.values,
                        marker=dict(
                            color=symptom_counts.values,
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=symptom_counts.values,
                        textposition='auto'
                    )
                ])

                fig.update_layout(
                    title=f"Top 15 Symptoms in {selected_state}",
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#2d3139', tickangle=-45),
                    yaxis=dict(gridcolor='#2d3139', title='Frequency'),
                    height=450
                )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # India-wide comparisons
            st.subheader("🇮🇳 All-India Comparisons")

            compare_tabs = st.tabs(["📊 Disease Rankings", "🗺️ State Rankings", "👥 Age & Gender Patterns"])

            with compare_tabs[0]:
                # Top diseases across India
                top_diseases_india_full = indian_df['Disease'].value_counts().head(15)

                fig = go.Figure(data=[
                    go.Bar(
                        x=top_diseases_india_full.index,
                        y=top_diseases_india_full.values,
                        marker=dict(
                            color=top_diseases_india_full.values,
                            colorscale='Reds',
                            showscale=True
                        ),
                        text=top_diseases_india_full.values,
                        textposition='auto'
                    )
                ])

                fig.update_layout(
                    title="Top 15 Diseases in India",
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#2d3139', tickangle=-45),
                    yaxis=dict(gridcolor='#2d3139', title='Cases'),
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

            with compare_tabs[1]:
                # State-wise case distribution
                state_counts = indian_df['State'].value_counts().head(20)

                fig = go.Figure(data=[
                    go.Bar(
                        x=state_counts.values,
                        y=state_counts.index,
                        orientation='h',
                        marker=dict(
                            color=state_counts.values,
                            colorscale='Blues',
                            showscale=True
                        ),
                        text=state_counts.values,
                        textposition='auto'
                    )
                ])

                fig.update_layout(
                    title="Top 20 States/UTs by Case Count",
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#2d3139', title='Cases'),
                    yaxis=dict(gridcolor='#2d3139'),
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

            with compare_tabs[2]:
                pattern_col1, pattern_col2 = st.columns(2)

                with pattern_col1:
                    # Age groups
                    indian_df['Age_Group'] = pd.cut(
                        indian_df['Age'],
                        bins=[0, 18, 30, 45, 60, 100],
                        labels=['<18', '18-30', '30-45', '45-60', '60+']
                    )

                    age_disease = indian_df.groupby(['Age_Group', 'Disease']).size().reset_index(name='Count')
                    # Filter out zero counts to avoid ZeroDivisionError
                    age_disease = age_disease[age_disease['Count'] > 0]
                    age_disease = age_disease.sort_values('Count', ascending=False).groupby('Age_Group').head(3)

                    if len(age_disease) > 0:
                        try:
                            fig = px.sunburst(
                                age_disease,
                                path=['Age_Group', 'Disease'],
                                values='Count',
                                title='Disease Distribution by Age Group',
                                color='Count',
                                color_continuous_scale='Reds'
                            )

                            fig.update_layout(
                                plot_bgcolor='#0e1117',
                                paper_bgcolor='#0e1117',
                                font=dict(color='white'),
                                height=500
                            )

                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            # Fallback to bar chart if sunburst fails
                            st.warning("Displaying as bar chart instead")
                            fig = px.bar(
                                age_disease,
                                x='Disease',
                                y='Count',
                                color='Age_Group',
                                title='Disease Distribution by Age Group',
                                barmode='group'
                            )
                            fig.update_layout(
                                plot_bgcolor='#0e1117',
                                paper_bgcolor='#0e1117',
                                font=dict(color='white'),
                                xaxis=dict(tickangle=-45),
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No age group data available")

                with pattern_col2:
                    # Gender patterns
                    gender_disease = indian_df.groupby(['Gender', 'Disease']).size().reset_index(name='Count')
                    gender_disease = gender_disease.sort_values('Count', ascending=False).groupby('Gender').head(5)

                    fig = px.bar(
                        gender_disease,
                        x='Disease',
                        y='Count',
                        color='Gender',
                        barmode='group',
                        title='Top 5 Diseases by Gender',
                        color_discrete_map={'M': '#00d9ff', 'F': '#ff006e'}
                    )

                    fig.update_layout(
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        xaxis=dict(gridcolor='#2d3139', tickangle=-45),
                        yaxis=dict(gridcolor='#2d3139'),
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 3: GLOBAL OVERVIEW
    # ========================================================================

    with tabs[2]:
        if deaths_df is None:
            st.error("❌ Death statistics data not available!")
        else:
            cause_cols = get_cause_columns(deaths_df)
            country_col = get_country_column(deaths_df)
            deaths_df = calculate_total_deaths(deaths_df, cause_cols)

            st.header("🌍 Global Health Overview")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("🌐 Countries", deaths_df[country_col].nunique())
            col2.metric("💀 Total Deaths", format_number(deaths_df['Total_Deaths'].sum()))
            col3.metric("📅 Year Range", f"{int(deaths_df['Year'].min())}-{int(deaths_df['Year'].max())}")

            top_causes = deaths_df[cause_cols].sum().sort_values(ascending=False)
            col4.metric("⚠️ Top Cause", top_causes.index[0][:20])

            st.markdown("---")

            # Top causes chart
            top_10_causes = top_causes.head(10)

            fig = go.Figure(data=[
                go.Bar(
                    x=top_10_causes.values,
                    y=top_10_causes.index,
                    orientation='h',
                    marker=dict(color=top_10_causes.values, colorscale='Reds', showscale=True),
                    text=[format_number(v) for v in top_10_causes.values],
                    textposition='auto',
                )
            ])

            fig.update_layout(
                title="Top 10 Causes of Death Globally",
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#2d3139', title='Total Deaths'),
                yaxis=dict(gridcolor='#2d3139', title=''),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Pie chart
            st.subheader("🥧 Disease Category Distribution")

            top_8 = top_causes.head(8)
            others_total = top_causes.sum() - top_8.sum()

            pie_data = pd.DataFrame({
                'Cause': list(top_8.index) + ['Others'],
                'Deaths': list(top_8.values) + [others_total]
            })

            fig = px.pie(
                pie_data,
                values='Deaths',
                names='Cause',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )

            fig.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 4: COUNTRY ANALYSIS
    # ========================================================================

    with tabs[3]:
        if deaths_df is None:
            st.error("❌ Data not available!")
        else:
            st.header("🗺️ Country-Wise Analysis")

            countries = sorted(deaths_df[country_col].unique())
            default_country = 'United States' if 'United States' in countries else countries[0]

            selected_country = st.selectbox(
                "🔍 Select a Country",
                countries,
                index=countries.index(default_country)
            )

            country_df = deaths_df[deaths_df[country_col] == selected_country]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("💀 Total Deaths", format_number(country_df['Total_Deaths'].sum()))
            col2.metric("📅 Years Covered", country_df['Year'].nunique())

            avg_yearly = country_df.groupby('Year')['Total_Deaths'].sum().mean()
            col3.metric("📊 Avg Deaths/Year", format_number(avg_yearly))

            country_top_cause = country_df[cause_cols].sum().idxmax()
            col4.metric("⚠️ Top Cause", country_top_cause[:20])

            st.markdown("---")

            # Top causes in country
            country_causes = country_df[cause_cols].sum().sort_values(ascending=False).head(10)

            fig = go.Figure(data=[
                go.Bar(
                    x=country_causes.index,
                    y=country_causes.values,
                    marker=dict(color=country_causes.values, colorscale='Blues'),
                    text=[format_number(v) for v in country_causes.values],
                    textposition='auto',
                )
            ])

            fig.update_layout(
                title=f"Top Causes in {selected_country}",
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#2d3139', title='', tickangle=-45),
                yaxis=dict(gridcolor='#2d3139', title='Total Deaths'),
                height=450
            )

            st.plotly_chart(fig, use_container_width=True)

            # Trend over years
            st.subheader(f"📈 Deaths Over Time in {selected_country}")

            yearly_country = country_df.groupby('Year')['Total_Deaths'].sum().reset_index()

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=yearly_country['Year'],
                y=yearly_country['Total_Deaths'],
                mode='lines+markers',
                line=dict(color='#00d9ff', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(0, 217, 255, 0.1)'
            ))

            fig.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#2d3139', title='Year'),
                yaxis=dict(gridcolor='#2d3139', title='Total Deaths'),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 5: DISEASE IMPACT (WITH CHOROPLETH HEATMAP)
    # ========================================================================

    with tabs[4]:
        if deaths_df is None:
            st.error("❌ Data not available!")
        else:
            st.header("🦠 Disease Impact Analysis")

            selected_disease = st.selectbox("🔍 Select a Disease/Cause", cause_cols)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("💀 Total Deaths", format_number(deaths_df[selected_disease].sum()))
            col2.metric("🌍 Affected Countries", deaths_df[deaths_df[selected_disease] > 0][country_col].nunique())
            col3.metric("📊 Avg per Country", format_number(deaths_df.groupby(country_col)[selected_disease].sum().mean()))
            col4.metric("📅 Peak Year", int(deaths_df.groupby('Year')[selected_disease].sum().idxmax()))

            st.markdown("---")

            # CHOROPLETH HEATMAP (NEW)
            st.subheader(f"🗺️ Global Heatmap: {selected_disease}")

            disease_by_country = deaths_df.groupby(country_col)[selected_disease].sum().reset_index()
            disease_by_country.columns = ['Country', 'Deaths']

            fig_map = px.choropleth(
                disease_by_country,
                locations='Country',
                locationmode='country names',
                color='Deaths',
                hover_name='Country',
                hover_data={'Deaths': ':,.0f'},
                color_continuous_scale='Reds',
                title=f'{selected_disease} - Global Distribution'
            )

            fig_map.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                geo=dict(
                    bgcolor='#0e1117',
                    showframe=False,
                    showcoastlines=True,
                    projection_type='natural earth',
                    landcolor='#2a2a2a',
                    oceancolor='#0e1117',
                    coastlinecolor='#555555'
                ),
                height=600,
                margin=dict(l=0, r=0, t=60, b=0)
            )

            st.plotly_chart(fig_map, use_container_width=True)

            st.markdown("---")

            # Most affected countries
            st.subheader(f"📊 Most Affected Countries")

            top_affected = deaths_df.groupby(country_col)[selected_disease].sum().sort_values(ascending=False).head(15)

            fig = go.Figure(data=[
                go.Bar(
                    x=top_affected.values,
                    y=top_affected.index,
                    orientation='h',
                    marker=dict(color=top_affected.values, colorscale='Oranges', showscale=True),
                    text=[format_number(v) for v in top_affected.values],
                    textposition='auto',
                )
            ])

            fig.update_layout(
                title=f"Top 15 Countries",
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#2d3139', title='Total Deaths'),
                yaxis=dict(gridcolor='#2d3139', title=''),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Trend over years
            st.subheader(f"📈 {selected_disease} Trend Over Years")

            yearly_disease = deaths_df.groupby('Year')[selected_disease].sum().reset_index()

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=yearly_disease['Year'],
                y=yearly_disease[selected_disease],
                mode='lines+markers',
                line=dict(color='#8338ec', width=4),
                marker=dict(size=10),
                fill='tozeroy',
                fillcolor='rgba(131, 56, 236, 0.2)'
            ))

            fig.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#2d3139', title='Year'),
                yaxis=dict(gridcolor='#2d3139', title='Deaths'),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 6: TIME SERIES
    # ========================================================================

    with tabs[5]:
        if deaths_df is None:
            st.error("❌ Data not available!")
        else:
            st.header("📈 Time Series Analysis")

            st.subheader("🔍 Multi-Cause Trend Comparison")

            num_causes = st.slider("Number of causes to compare", 3, 10, 5)
            top_causes_list = deaths_df[cause_cols].sum().sort_values(ascending=False).head(num_causes).index.tolist()

            fig = go.Figure()

            colors = ['#00d9ff', '#ff006e', '#8338ec', '#ffbe0b', '#fb5607',
                      '#06ffa5', '#ff9e00', '#9d4edd', '#06ffa5', '#e63946']

            for i, cause in enumerate(top_causes_list):
                yearly_cause = deaths_df.groupby('Year')[cause].sum().reset_index()

                fig.add_trace(go.Scatter(
                    x=yearly_cause['Year'],
                    y=yearly_cause[cause],
                    mode='lines+markers',
                    name=cause,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6)
                ))

            fig.update_layout(
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#2d3139', title='Year'),
                yaxis=dict(gridcolor='#2d3139', title='Deaths'),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                height=600,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🏥 Integrated Health Analytics Dashboard | Streamlit + Plotly + Scikit-learn</p>
            <p>Data: Kaggle + Patient Symptoms + India Disease Dataset</p>
            <p>© 2025 AI-Powered Health Analytics</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
