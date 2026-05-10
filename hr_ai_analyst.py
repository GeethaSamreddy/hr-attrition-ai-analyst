import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from io import StringIO
from dotenv import load_dotenv
import os
load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition AI Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.main { background: #0d0d0f; }
.block-container { padding: 2rem 2.5rem; }

.kpi-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.kpi-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #a78bfa; }
.kpi-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
.kpi-delta { font-size: 0.8rem; margin-top: 6px; }

.chat-bubble-user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 0;
    margin-left: 20%;
    font-size: 0.9rem;
}
.chat-bubble-ai {
    background: #1e1e2e;
    border: 1px solid rgba(99,102,241,0.25);
    color: #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 0;
    margin-right: 10%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid rgba(167,139,250,0.2);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.insight-card {
    background: #1e1e2e;
    border-left: 3px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    color: #cbd5e1;
    font-size: 0.88rem;
    line-height: 1.6;
}
.stTextInput > div > div > input {
    background: #1e1e2e !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file_bytes):
    df = pd.read_csv(StringIO(file_bytes.decode("utf-8")))
    df.columns = df.columns.str.strip().str.lstrip('\ufeff')
    df['Attrition_Flag'] = (df['Attrition'] == 'Yes').astype(int)
    return df

def get_data_summary(df):
    """Create a compact summary of the dataframe for the LLM."""
    attrition_rate = df['Attrition_Flag'].mean() * 100
    dept_attrition = df.groupby('Department')['Attrition_Flag'].mean().mul(100).round(1).to_dict()
    role_attrition = df.groupby('JobRole')['Attrition_Flag'].mean().mul(100).round(1).nlargest(5).to_dict()
    avg_income_left = df[df['Attrition']=='Yes']['MonthlyIncome'].mean()
    avg_income_stayed = df[df['Attrition']=='No']['MonthlyIncome'].mean()
    overtime_attrition = df.groupby('OverTime')['Attrition_Flag'].mean().mul(100).round(1).to_dict()
    age_attrition = df.groupby(pd.cut(df['Age'], bins=[18,25,35,45,60]))['Attrition_Flag'].mean().mul(100).round(1).to_dict()
    age_attrition = {str(k): v for k, v in age_attrition.items()}

    return f"""
HR Dataset Summary (1,470 employees):
- Overall attrition rate: {attrition_rate:.1f}%
- Total employees: {len(df)}, Left: {df['Attrition_Flag'].sum()}, Stayed: {len(df) - df['Attrition_Flag'].sum()}
- Attrition by Department: {dept_attrition}
- Top 5 high-attrition Job Roles: {role_attrition}
- Avg Monthly Income (Left): ${avg_income_left:,.0f} | (Stayed): ${avg_income_stayed:,.0f}
- Overtime attrition: {overtime_attrition}
- Attrition by Age Group: {age_attrition}
- Avg Age: {df['Age'].mean():.1f} | Avg Years at Company: {df['YearsAtCompany'].mean():.1f}
- Gender split: {df['Gender'].value_counts().to_dict()}
- Marital status attrition: {df.groupby('MaritalStatus')['Attrition_Flag'].mean().mul(100).round(1).to_dict()}
"""

def ask_ai(question, df, history):
    """Send question to Groq with data context."""
    client = Groq()
    data_summary = get_data_summary(df)

    system_prompt = f"""You are an expert HR Data Analyst AI. You have access to an HR attrition dataset.

{data_summary}

Answer questions about this HR data concisely and insightfully. 
- Give specific numbers from the data
- Highlight business implications
- Keep answers to 3-5 sentences max unless asked for more
- Be direct and confident like a senior analyst presenting to a CEO
- If asked for a chart suggestion, mention what chart type would show this best"""

    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=600,
        messages=messages
    )
    return response.choices[0].message.content

def generate_executive_summary(df):
    """Generate a full executive summary using Groq."""
    client = Groq()
    data_summary = get_data_summary(df)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",       
        max_tokens=1000,
        messages=[
            {"role": "system", "content": "You are a senior HR consultant writing for a C-suite audience. Be specific, data-driven, and actionable."},
            {"role": "user", "content": f"""Based on this HR data, write a professional executive summary for the HR Director.
            
{data_summary}

Format as:
1. Key Finding (1 sentence headline)
2. Critical Risk Areas (3 bullet points with specific numbers)
3. Root Causes (2-3 bullet points)
4. Recommended Actions (3 bullet points)
5. Business Impact (1-2 sentences on cost/risk)

Be specific with numbers. Write like a McKinsey consultant."""}
        ]
    )
    return response.choices[0].message.content

# ── Charts ────────────────────────────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='DM Sans'),
    margin=dict(l=10, r=10, t=30, b=10)
)

def chart_dept_attrition(df):
    data = df.groupby('Department')['Attrition_Flag'].mean().mul(100).round(1).reset_index()
    data.columns = ['Department', 'Attrition %']
    fig = px.bar(data, x='Department', y='Attrition %', color='Attrition %',
                 color_continuous_scale=['#4f46e5','#a78bfa','#f472b6'],
                 title='Attrition Rate by Department')
    fig.update_layout(**CHART_THEME)
    fig.update_coloraxes(showscale=False)
    return fig

def chart_income_attrition(df):
    fig = px.box(df, x='Attrition', y='MonthlyIncome', color='Attrition',
                 color_discrete_map={'Yes':'#f472b6','No':'#6366f1'},
                 title='Monthly Income vs Attrition')
    fig.update_layout(**CHART_THEME)
    return fig

def chart_overtime(df):
    data = df.groupby(['OverTime','Attrition']).size().reset_index(name='Count')
    fig = px.bar(data, x='OverTime', y='Count', color='Attrition',
                 color_discrete_map={'Yes':'#f472b6','No':'#6366f1'},
                 barmode='group', title='Overtime vs Attrition')
    fig.update_layout(**CHART_THEME)
    return fig

def chart_age_attrition(df):
    df2 = df.copy()
    df2['AgeGroup'] = pd.cut(df2['Age'], bins=[18,25,35,45,60],
                              labels=['18-25','26-35','36-45','46-60'])
    data = df2.groupby('AgeGroup', observed=True)['Attrition_Flag'].mean().mul(100).round(1).reset_index()
    fig = px.line(data, x='AgeGroup', y='Attrition_Flag', markers=True,
                  title='Attrition Rate by Age Group',
                  color_discrete_sequence=['#a78bfa'])
    fig.update_layout(**CHART_THEME)
    return fig

def chart_role_attrition(df):
    data = df.groupby('JobRole')['Attrition_Flag'].mean().mul(100).round(1).reset_index()
    data.columns = ['JobRole','Attrition %']
    data = data.sort_values('Attrition %', ascending=True)
    fig = px.bar(data, x='Attrition %', y='JobRole', orientation='h',
                 color='Attrition %', color_continuous_scale=['#4f46e5','#f472b6'],
                 title='Attrition by Job Role')
    fig.update_layout(**CHART_THEME, height=350)
    fig.update_coloraxes(showscale=False)
    return fig

# ── App Layout ────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-family:Syne;color:#a78bfa;font-size:2rem;margin-bottom:0'>🧠 HR Attrition AI Analyst</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b;margin-top:4px;margin-bottom:2rem'>Ask anything about your HR data — powered by Claude AI</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='section-header'>📂 Data</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload HR CSV", type=["csv"])
    st.markdown("---")
    st.markdown("<div class='section-header'>💡 Try asking</div>", unsafe_allow_html=True)
    sample_questions = [
        "Which department has highest attrition?",
        "Do employees who work overtime leave more?",
        "What is the salary gap between who left and stayed?",
        "Which age group is at highest risk?",
        "What are the top 3 reasons employees leave?",
        "Which job roles need urgent attention?"
    ]
    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state['prefill'] = q

# Load data
if uploaded:
    df = load_data(uploaded.read())
else:
    # Try loading default
    try:
        df = pd.read_csv("/app/HR-Em.csv")
        df.columns = df.columns.str.strip().str.lstrip('\ufeff')
        df['Attrition_Flag'] = (df['Attrition'] == 'Yes').astype(int)
        st.info("Using default HR dataset. Upload your own CSV in the sidebar.")
    except:
        st.warning("👈 Please upload your HR CSV file in the sidebar to get started.")
        st.stop()

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)
attrition_rate = df['Attrition_Flag'].mean() * 100
total_left = df['Attrition_Flag'].sum()
avg_income_left = df[df['Attrition']=='Yes']['MonthlyIncome'].mean()
avg_income_stayed = df[df['Attrition']=='No']['MonthlyIncome'].mean()
income_gap = avg_income_stayed - avg_income_left
overtime_risk = df[df['OverTime']=='Yes']['Attrition_Flag'].mean() * 100

with col1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{attrition_rate:.1f}%</div><div class='kpi-label'>Attrition Rate</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_left}</div><div class='kpi-label'>Employees Left</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>${avg_income_left:,.0f}</div><div class='kpi-label'>Avg Income (Left)</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>${income_gap:,.0f}</div><div class='kpi-label'>Income Gap</div></div>", unsafe_allow_html=True)
with col5:
    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{overtime_risk:.1f}%</div><div class='kpi-label'>Overtime Attrition</div></div>", unsafe_allow_html=True)

st.markdown("---")

# Two column layout: Chat | Charts
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='section-header'>💬 Ask AI Analyst</div>", unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_history' not in st.session_state:
        st.session_state.api_history = []

    # Chat display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("<div class='chat-bubble-ai'>👋 Hi! I'm your HR AI Analyst. Ask me anything about this dataset — attrition rates, risk factors, salary gaps, or what actions to take.</div>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble-ai'>{msg['content']}</div>", unsafe_allow_html=True)

    # Input
    prefill = st.session_state.pop('prefill', '')
    question = st.text_input("Ask a question...", value=prefill, key="q_input", label_visibility="collapsed", placeholder="e.g. Which department has highest attrition?")

    col_ask, col_clear = st.columns([3,1])
    with col_ask:
        ask_clicked = st.button("Ask AI ✦", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.api_history = []
            st.rerun()

    if ask_clicked and question.strip():
        with st.spinner("Analysing..."):
            answer = ask_ai(question, df, st.session_state.api_history)
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.api_history.append({"role": "user", "content": question})
        st.session_state.api_history.append({"role": "assistant", "content": answer})
        st.rerun()

    st.markdown("---")
    st.markdown("<div class='section-header'>📋 Executive Summary</div>", unsafe_allow_html=True)
    if st.button("Generate Executive Summary ✦", use_container_width=True):
        with st.spinner("Writing executive summary..."):
            summary = generate_executive_summary(df)
        for line in summary.split('\n'):
            if line.strip():
                st.markdown(f"<div class='insight-card'>{line}</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='section-header'>📊 Visual Analytics</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Department", "Income", "Overtime", "Age", "Job Role"])
    with tab1:
        st.plotly_chart(chart_dept_attrition(df), use_container_width=True)
    with tab2:
        st.plotly_chart(chart_income_attrition(df), use_container_width=True)
    with tab3:
        st.plotly_chart(chart_overtime(df), use_container_width=True)
    with tab4:
        st.plotly_chart(chart_age_attrition(df), use_container_width=True)
    with tab5:
        st.plotly_chart(chart_role_attrition(df), use_container_width=True)