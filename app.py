"""
AI DataChat - Complete Application with Subscriptions and Monetization
Modern UI Version - All Features Preserved + Sales Pipeline Analyzer

AI DataChat Proprietary License
© 2025 Gardel Hiram. All rights reserved.

Permission is hereby granted to view and test this software for evaluation purposes only.
Any reproduction, modification, distribution, or commercial use of this software,
in whole or in part, without the express written consent of Gardel Hiram,
is strictly prohibited.

For licensing inquiries or permission requests, contact: gardelhiram9@gmail.com
"""
import streamlit as st

# MUST BE FIRST
st.set_page_config(
    page_title="AI DataChat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
import os

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.stripe_handler import StripeHandler
from src.utils.subscription import SubscriptionManager
from src.utils.file_handler import FileHandler
from src.utils.data_analyzer import DataAnalyzer
from src.agents.ai_agent import AIAgent
from src.utils.auth import AuthManager
from src.utils.data_quality import DataQualityAnalyzer
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_custom_styles():
    """Apply modern dark theme with glassmorphism"""
    st.markdown("""
    <style>
        /* Main App Background */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        }
        
        /* Hide default menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Headers */
        .main-header {
            font-size: 3rem;
            font-weight: 300;
            background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }
        
        .tagline {
            font-size: 1.3rem;
            color: #94a3b8;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Free Pro Banner */
        .free-pro-banner {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .free-pro-banner-text {
            color: #10b981;
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .free-pro-banner-subtitle {
            color: #6ee7b7;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.95);
            border-right: 1px solid rgba(6, 182, 212, 0.3);
            backdrop-filter: blur(10px);
        }
        
        [data-testid="stSidebar"] .element-container {
            color: #e5e7eb;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.75rem 2rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(31, 41, 55, 0.5);
            padding: 0.5rem;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(55, 65, 81, 0.5);
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 8px;
            font-weight: 600;
            color: #9ca3af !important;
            font-size: 1rem !important;
            background: transparent;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(6, 182, 212, 0.1);
            color: #22d3ee !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
        }
        
        .stTabs [aria-selected="true"] button {
            color: white !important;
        }
        
        /* Metrics */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(6, 182, 212, 0.3);
            backdrop-filter: blur(10px);
        }
        
        [data-testid="stMetricLabel"] {
            color: #9ca3af !important;
            font-size: 0.875rem;
            font-weight: 400;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 300;
            color: #22d3ee !important;
        }
        
        /* User Info Badge */
        .user-info {
            text-align: right;
            padding: 0.75rem 1rem;
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
            color: #e5e7eb;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(6, 182, 212, 0.2);
            border: 1px solid rgba(6, 182, 212, 0.3);
            backdrop-filter: blur(10px);
        }
        
        /* Pro Badge */
        .pro-badge {
            display: inline-block;
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-left: 0.5rem;
            box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: rgba(31, 41, 55, 0.5);
            border: 2px dashed rgba(6, 182, 212, 0.3);
            border-radius: 12px;
            padding: 2rem;
            backdrop-filter: blur(10px);
        }
        
        /* Selectbox and Input */
        .stSelectbox > div > div, .stTextInput > div > div > input {
            background: rgba(55, 65, 81, 0.8) !important;
            border: 1px solid rgba(75, 85, 99, 0.5) !important;
            color: #e5e7eb !important;
            border-radius: 8px;
        }
        
        /* Dataframe */
        [data-testid="stDataFrame"] {
            background: rgba(31, 41, 55, 0.5);
            border-radius: 12px;
            border: 1px solid rgba(55, 65, 81, 0.5);
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: rgba(31, 41, 55, 0.5);
            border-radius: 8px;
            color: #e5e7eb !important;
            border: 1px solid rgba(55, 65, 81, 0.5);
        }
        
        /* Text and Labels */
        p, label, .stMarkdown {
            color: #d1d5db !important;
        }
        
        h1, h2, h3, h4 {
            color: #e5e7eb !important;
            font-weight: 300 !important;
        }
        
        /* Progress bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%);
        }
        
        /* Info/Warning/Error boxes */
        .stAlert {
            background: rgba(31, 41, 55, 0.8);
            border-radius: 12px;
            border-left: 4px solid;
            backdrop-filter: blur(10px);
        }
        
        /* Chat messages */
        [data-testid="stChatMessage"] {
            background: rgba(31, 41, 55, 0.5);
            border-radius: 12px;
            border: 1px solid rgba(55, 65, 81, 0.5);
            backdrop-filter: blur(10px);
        }
        
        /* Plotly charts container */
        .js-plotly-plot {
            border-radius: 12px;
            background: rgba(31, 41, 55, 0.5);
            padding: 1rem;
            border: 1px solid rgba(55, 65, 81, 0.5);
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_filtered' not in st.session_state:
        st.session_state.df_filtered = None
    if 'file_info' not in st.session_state:
        st.session_state.file_info = None
    if 'data_summary' not in st.session_state:
        st.session_state.data_summary = None
    if 'ai_agent' not in st.session_state:
        st.session_state.ai_agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'narrative_report' not in st.session_state:
        st.session_state.narrative_report = None
    if 'html_report' not in st.session_state:
        st.session_state.html_report = None
    if 'show_pricing' not in st.session_state:
        st.session_state.show_pricing = False
    if 'is_sales_data' not in st.session_state:
        st.session_state.is_sales_data = False


def detect_sales_dataset(df):
    """Detect if uploaded dataset is sales-related"""
    if df is None:
        return False
    
    sales_keywords = [
        'deal', 'stage', 'pipeline', 'close', 'amount', 'value', 'revenue',
        'opportunity', 'lead', 'prospect', 'win', 'lost', 'owner', 'sales',
        'customer', 'contract', 'proposal', 'qualified', 'negotiation'
    ]
    
    column_names = [col.lower() for col in df.columns]
    
    matches = sum(1 for keyword in sales_keywords for col in column_names if keyword in col)
    
    return matches >= 3


def show_free_pro_banner():
    """Display banner announcing free Pro features"""
    st.markdown("""
    <div class="free-pro-banner">
        <div style="font-size: 2rem;">🎉</div>
        <div style="flex: 1;">
            <div class="free-pro-banner-text">All Pro Features Currently Free!</div>
            <div class="free-pro-banner-subtitle">No payment required - Test all premium features including Sales Pipeline Analyzer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def login_page():
    """Display login page with modern styling"""
    st.markdown('<h1 class="main-header">AI DataChat</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Universal Intelligence Through Data</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("Authentication")
        
        auth_manager = AuthManager()
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if username and password:
                        if auth_manager.authenticate(username, password):
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.user_info = auth_manager.get_user_info(username)
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    else:
                        st.warning("Please enter both username and password")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Choose Username")
                new_name = st.text_input("Full Name")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register = st.form_submit_button("Register", use_container_width=True)
                
                if register:
                    if not new_username or not new_password or not new_name:
                        st.error("All fields are required")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        if auth_manager.register_user(new_username, new_password, new_name):
                            st.success("Registration successful! Please login.")
                        else:
                            st.error("Username already exists")


def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_info = None
    st.session_state.df = None
    st.session_state.df_filtered = None
    st.session_state.file_info = None
    st.session_state.data_summary = None
    st.session_state.ai_agent = None
    st.session_state.chat_history = []
    st.session_state.narrative_report = None
    st.session_state.html_report = None
    st.session_state.show_pricing = False
    st.session_state.is_sales_data = False
    st.rerun()


def load_data_file(uploaded_file):
    """Load and process uploaded file"""
    import tempfile
    
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / f"temp_{uploaded_file.name}"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = FileHandler.load_file(temp_path)
        file_info = FileHandler.get_file_info(temp_path)
        
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()
        
        data_summary = DataAnalyzer.get_summary_statistics(df)
        
        st.session_state.df = df
        st.session_state.df_filtered = df.copy()
        st.session_state.file_info = file_info
        st.session_state.data_summary = data_summary
        
        # Detect if this is sales data
        st.session_state.is_sales_data = detect_sales_dataset(df)
        
        try:
            st.session_state.ai_agent = AIAgent()
        except Exception as e:
            logger.warning(f"Could not initialize AI agent: {str(e)}")
            st.session_state.ai_agent = None
        
        logger.info(f"Successfully loaded file: {uploaded_file.name}")
        return True
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        logger.error(f"Error loading file: {str(e)}")
        return False


def sales_pipeline_analyzer():
    """Sales Pipeline Analyzer interface"""
    st.header("💰 Sales Pipeline Analyzer")
    
    show_free_pro_banner()
    
    if st.session_state.df_filtered is None:
        st.info("Please upload sales data to begin analysis")
        st.markdown("""
        ### Expected Data Format
        Your CSV should include columns such as:
        - **Deal ID** or **Opportunity ID**
        - **Stage** (e.g., Qualified, Proposal, Negotiation, Closed)
        - **Amount** or **Value**
        - **Close Date**
        - **Owner** or **Sales Rep**
        - **Probability** (optional)
        """)
        return
    
    df = st.session_state.df_filtered
    
    # Detect sales-related columns
    stage_col = None
    amount_col = None
    date_col = None
    owner_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'stage' in col_lower and stage_col is None:
            stage_col = col
        elif any(keyword in col_lower for keyword in ['amount', 'value', 'revenue']) and amount_col is None:
            amount_col = col
        elif any(keyword in col_lower for keyword in ['close', 'date']) and date_col is None:
            date_col = col
        elif any(keyword in col_lower for keyword in ['owner', 'rep', 'sales']) and owner_col is None:
            owner_col = col
    
    st.subheader("Configure Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stage_column = st.selectbox(
            "Sales Stage Column:",
            df.columns.tolist(),
            index=df.columns.tolist().index(stage_col) if stage_col else 0,
            key="sales_stage_col"
        )
        
        amount_column = st.selectbox(
            "Deal Amount Column:",
            df.select_dtypes(include=[np.number]).columns.tolist(),
            index=df.select_dtypes(include=[np.number]).columns.tolist().index(amount_col) if amount_col and amount_col in df.select_dtypes(include=[np.number]).columns else 0,
            key="sales_amount_col"
        )
    
    with col2:
        date_column = st.selectbox(
            "Close Date Column (optional):",
            ["None"] + df.columns.tolist(),
            index=df.columns.tolist().index(date_col) + 1 if date_col else 0,
            key="sales_date_col"
        )
        
        owner_column = st.selectbox(
            "Deal Owner Column (optional):",
            ["None"] + df.columns.tolist(),
            index=df.columns.tolist().index(owner_col) + 1 if owner_col else 0,
            key="sales_owner_col"
        )
    
    if st.button("📊 Generate Pipeline Analysis", type="primary", key="generate_sales_analysis"):
        with st.spinner("Analyzing sales pipeline..."):
            try:
                # Calculate pipeline metrics
                stages = df[stage_column].unique()
                
                st.markdown("---")
                st.subheader("Sales Pipeline Overview")
                
                # Key metrics
                total_deals = len(df)
                total_value = df[amount_column].sum()
                
                closed_stages = [s for s in stages if 'close' in str(s).lower() or 'won' in str(s).lower()]
                if closed_stages:
                    closed_df = df[df[stage_column].isin(closed_stages)]
                    closed_deals = len(closed_df)
                    closed_value = closed_df[amount_column].sum()
                    win_rate = (closed_deals / total_deals * 100) if total_deals > 0 else 0
                else:
                    closed_deals = 0
                    closed_value = 0
                    win_rate = 0
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric("Total Deals", f"{total_deals:,}")
                with col_m2:
                    st.metric("Total Pipeline Value", f"${total_value:,.0f}")
                with col_m3:
                    st.metric("Closed Value", f"${closed_value:,.0f}")
                with col_m4:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                st.markdown("---")
                
                # Pipeline Funnel Visualization
                col_viz1, col_viz2 = st.columns([2, 1])
                
                with col_viz1:
                    st.subheader("Pipeline Funnel")
                    
                    stage_summary = df.groupby(stage_column)[amount_column].agg(['count', 'sum']).reset_index()
                    stage_summary.columns = ['Stage', 'Deal Count', 'Total Value']
                    stage_summary = stage_summary.sort_values('Total Value', ascending=False)
                    
                    # Create funnel chart
                    fig_funnel = go.Figure()
                    
                    colors = ['#3b82f6', '#2563eb', '#1e40af', '#1e3a8a', '#172554']
                    
                    for idx, row in stage_summary.iterrows():
                        fig_funnel.add_trace(go.Funnel(
                            name=row['Stage'],
                            y=[row['Stage']],
                            x=[row['Total Value']],
                            textinfo="value+percent initial",
                            marker=dict(color=colors[idx % len(colors)])
                        ))
                    
                    fig_funnel.update_layout(
                        title="Sales Funnel by Value",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e5e7eb'),
                        height=500
                    )
                    
                    st.plotly_chart(fig_funnel, use_container_width=True)
                    
                    # Stage breakdown table
                    st.dataframe(stage_summary, use_container_width=True, hide_index=True)
                
                with col_viz2:
                    # Trust Score Calculation
                    st.subheader("Data Quality Score")
                    
                    score = 100
                    issues = []
                    
                    # Check for missing dates
                    if date_column != "None":
                        missing_dates = df[date_column].isna().sum()
                        if missing_dates > 0:
                            penalty = min(20, (missing_dates / len(df)) * 30)
                            score -= penalty
                            issues.append(f"Missing close dates: {missing_dates}")
                    
                    # Check for missing owners
                    if owner_column != "None":
                        missing_owners = df[owner_column].isna().sum()
                        if missing_owners > 0:
                            penalty = min(15, (missing_owners / len(df)) * 25)
                            score -= penalty
                            issues.append(f"Missing deal owners: {missing_owners}")
                    
                    # Check for zero/negative amounts
                    invalid_amounts = (df[amount_column] <= 0).sum()
                    if invalid_amounts > 0:
                        penalty = min(15, (invalid_amounts / len(df)) * 20)
                        score -= penalty
                        issues.append(f"Invalid amounts: {invalid_amounts}")
                    
                    score = max(0, int(score))
                    
                    # Display trust score
                    if score >= 80:
                        color = "#10b981"
                        grade = "Excellent"
                    elif score >= 60:
                        color = "#f59e0b"
                        grade = "Good"
                    else:
                        color = "#ef4444"
                        grade = "Needs Improvement"
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%); border-radius: 12px; border: 1px solid rgba(6, 182, 212, 0.3);'>
                        <h1 style='color: {color}; font-size: 3.5rem; margin: 0;'>{score}</h1>
                        <p style='font-size: 1rem; color: #9ca3af; margin: 5px 0;'>Data Quality</p>
                        <p style='font-size: 0.9rem; color: #6b7280;'>{grade}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if issues:
                        st.markdown("---")
                        st.warning("**Issues Detected:**")
                        for issue in issues:
                            st.write(f"• {issue}")
                    
                    # Forecast
                    st.markdown("---")
                    st.subheader("Forecast")
                    
                    open_stages = [s for s in stages if s not in closed_stages]
                    if open_stages:
                        forecast_df = df[df[stage_column].isin(open_stages)]
                        forecast_value = forecast_df[amount_column].sum()
                        
                        st.metric("Expected Revenue", f"${forecast_value:,.0f}")
                        st.caption(f"From {len(forecast_df)} open deals")
                
                # Conversion Analysis
                st.markdown("---")
                st.subheader("Stage Conversion Analysis")
                
                stage_list = stage_summary['Stage'].tolist()
                
                if len(stage_list) >= 2:
                    conversions = []
                    
                    for i in range(len(stage_list) - 1):
                        from_stage = stage_list[i]
                        to_stage = stage_list[i + 1]
                        
                        from_count = stage_summary[stage_summary['Stage'] == from_stage]['Deal Count'].values[0]
                        to_count = stage_summary[stage_summary['Stage'] == to_stage]['Deal Count'].values[0]
                        
                        conversion_rate = (to_count / from_count * 100) if from_count > 0 else 0
                        
                        conversions.append({
                            'From': from_stage,
                            'To': to_stage,
                            'Conversion Rate': f"{conversion_rate:.1f}%",
                            'Deals Lost': from_count - to_count
                        })
                    
                    conversion_df = pd.DataFrame(conversions)
                    st.dataframe(conversion_df, use_container_width=True, hide_index=True)
                
                # Deal Distribution by Owner
                if owner_column != "None":
                    st.markdown("---")
                    st.subheader("Performance by Owner")
                    
                    owner_summary = df.groupby(owner_column)[amount_column].agg(['count', 'sum', 'mean']).reset_index()
                    owner_summary.columns = ['Owner', 'Deal Count', 'Total Value', 'Avg Deal Size']
                    owner_summary = owner_summary.sort_values('Total Value', ascending=False)
                    
                    fig_owner = px.bar(
                        owner_summary.head(10),
                        x='Owner',
                        y='Total Value',
                        title="Top 10 Performers by Pipeline Value",
                        color='Total Value',
                        color_continuous_scale='Teal'
                    )
                    
                    fig_owner.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e5e7eb'),
                        xaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                        yaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                        height=400
                    )
                    
                    st.plotly_chart(fig_owner, use_container_width=True)
                
                # Narrative Report
                st.markdown("---")
                st.subheader("💡 Key Insights")
                
                insights = []
                
                # Win rate insight
                if win_rate < 15:
                    insights.append(f"⚠️ Win rate is {win_rate:.1f}%, which is below industry average (typically 15-20%). Consider reviewing qualification criteria.")
                elif win_rate > 25:
                    insights.append(f"✅ Win rate of {win_rate:.1f}% is excellent, above industry average!")
                
                # Pipeline health
                if len(stage_list) >= 2:
                    top_stage_count = stage_summary.iloc[0]['Deal Count']
                    bottom_stage_count = stage_summary.iloc[-1]['Deal Count']
                    drop_rate = ((top_stage_count - bottom_stage_count) / top_stage_count * 100)
                    
                    if drop_rate > 70:
                        insights.append(f"⚠️ High drop-off rate of {drop_rate:.1f}% through the pipeline. Focus on conversion optimization.")
                
                # Data quality
                if score < 70:
                    insights.append(f"⚠️ Data quality score is {score}/100. Clean data will improve forecast accuracy.")
                
                for insight in insights:
                    st.info(insight)
                
                # Export
                st.markdown("---")
                st.subheader("📥 Export Analysis")
                
                export_data = {
                    'Stage Summary': stage_summary,
                    'Conversion Analysis': conversion_df if len(stage_list) >= 2 else pd.DataFrame()
                }
                
                if owner_column != "None":
                    export_data['Owner Performance'] = owner_summary
                
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    for sheet_name, data in export_data.items():
                        if not data.empty:
                            data.to_excel(writer, index=False, sheet_name=sheet_name)
                
                st.download_button(
                    label="📊 Download Full Analysis (Excel)",
                    data=buffer.getvalue(),
                    file_name=f"sales_pipeline_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_sales_analysis"
                )
                
            except Exception as e:
                st.error(f"Error analyzing pipeline: {str(e)}")
                logger.error(f"Sales analysis error: {str(e)}")


def clean_data_interface():
    """Improved data cleaning interface with better error handling"""
    st.subheader("Data Cleaning")
    
    if st.session_state.df is None:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Handle Missing Values**")
        missing_action = st.selectbox(
            "Action for missing values:",
            ["Keep as is", "Drop rows with any missing", "Drop rows with all missing", "Fill with mean", "Fill with median", "Fill with mode"],
            key="missing_action_selector"
        )
        
        if st.button("Apply Missing Value Treatment", key="apply_missing_button"):
            work = st.session_state.df_filtered.copy()
            numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
            
            if missing_action == "Drop rows with any missing":
                work = work.dropna()
            elif missing_action == "Drop rows with all missing":
                work = work.dropna(how='all')
            elif missing_action == "Fill with mean":
                for c in numeric_cols:
                    work[c] = work[c].fillna(work[c].mean())
            elif missing_action == "Fill with median":
                for c in numeric_cols:
                    work[c] = work[c].fillna(work[c].median())
            elif missing_action == "Fill with mode":
                for c in work.columns:
                    mode = work[c].mode(dropna=True)
                    if not mode.empty:
                        work[c] = work[c].fillna(mode.iloc[0])
            
            st.session_state.df_filtered = work
            st.success(f"Applied: {missing_action}")
            st.rerun()
    
    with col2:
        st.write("**Handle Outliers (Z-Score)**")
        numeric_cols = st.session_state.df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            outlier_col = st.selectbox("Select column:", numeric_cols, key="outlier_column_selector")
            
            if st.button("Remove Outliers (|z|>3)", key="remove_outliers_button"):
                work = st.session_state.df_filtered.copy()
                
                std = work[outlier_col].std()
                if std == 0 or np.isnan(std):
                    st.warning(f"No variability in '{outlier_col}' – no outliers to remove.")
                else:
                    z = np.abs((work[outlier_col] - work[outlier_col].mean()) / std)
                    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                    before_count = len(work)
                    work = work[z < 3]
                    removed_count = before_count - len(work)
                    
                    st.session_state.df_filtered = work
                    st.success(f"Removed {removed_count} outliers from {outlier_col}")
                    st.rerun()


def data_filtering_interface():
    """Advanced data filtering"""
    st.subheader("🔍 Filter Data")
    
    if st.session_state.df is None:
        return
    
    df = st.session_state.df.copy()
    
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to display:", 
        all_columns, 
        default=all_columns,
        key="column_filter_selector"
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        with st.expander("Numeric Filters"):
            for idx, col in enumerate(numeric_cols[:3]):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                selected_range = st.slider(
                    f"{col}:",
                    min_val, max_val, (min_val, max_val),
                    key=f"numeric_slider_{idx}_{col.replace(' ', '_')}"
                )
                df = df[(df[col] >= selected_range[0]) & (df[col] <= selected_range[1])]
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        with st.expander("Categorical Filters"):
            for idx, col in enumerate(cat_cols[:2]):
                unique_vals = df[col].unique().tolist()
                selected_vals = st.multiselect(
                    f"{col}:", 
                    unique_vals, 
                    default=unique_vals,
                    key=f"categorical_filter_{idx}_{col.replace(' ', '_')}"
                )
                df = df[df[col].isin(selected_vals)]
    
    if selected_columns:
        df = df[selected_columns]
    
    st.session_state.df_filtered = df
    st.info(f"Filtered: {len(df)} rows × {len(df.columns)} columns")


def create_visualizations(df):
    """Create interactive visualizations with modern color scheme"""
    st.subheader("Interactive Visualizations")
    
    if len(df) > 10000:
        st.warning(f"Large dataset detected ({len(df):,} rows). Some visualizations will use sampling for performance.")
    
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Histogram", "Scatter Plot", "Bar Chart", "Box Plot", "Correlation Heatmap", "Line Chart"],
        key="viz_type_selector"
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Modern color palette
    color_sequence = ['#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
    
    max_cols_for_viz = 20
    if len(numeric_cols) > max_cols_for_viz:
        st.info(f"Dataset has {len(numeric_cols)} numeric columns. Selecting top {max_cols_for_viz} most variable for performance.")
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(max_cols_for_viz).index.tolist()
    
    if viz_type == "Histogram":
        if numeric_cols:
            col = st.selectbox("Select column:", numeric_cols, key="hist_column")
            st.info(f"Showing distribution of {col}. Look for patterns, outliers, or skewness.")
            
            plot_df = df if len(df) <= 10000 else df.sample(10000)
            if len(df) > 10000:
                st.caption(f"Showing sample of 10,000 from {len(df):,} total rows")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=plot_df[col],
                nbinsx=50,
                marker=dict(
                    color='#06b6d4',
                    line=dict(color='#0891b2', width=1)
                ),
                name=col
            ))
            
            fig.update_layout(
                title=f"Distribution of {col}",
                xaxis_title=col,
                yaxis_title="Count",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                xaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                yaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{df[col].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[col].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df[col].std():.2f}")
    
    elif viz_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            st.info("Use scatter plots to identify relationships between two numeric variables.")
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
            with col2:
                y_col = st.selectbox("Y-axis:", [c for c in numeric_cols if c != x_col], key="scatter_y")
            
            color_options = ["None"] + cat_cols[:5] + numeric_cols[:5]
            color_col = st.selectbox("Color by (optional):", color_options, key="scatter_color")
            color_col = None if color_col == "None" else color_col
            
            sample_size = min(5000, len(df))
            plot_df = df.sample(sample_size) if len(df) > sample_size else df
            if len(df) > sample_size:
                st.caption(f"Showing sample of {sample_size:,} from {len(df):,} total rows")
            
            correlation = df[[x_col, y_col]].corr().iloc[0, 1]
            
            fig = px.scatter(
                plot_df, x=x_col, y=y_col, color=color_col,
                title=f"{x_col} vs {y_col} (Correlation: {correlation:.3f})",
                opacity=0.6,
                color_discrete_sequence=color_sequence
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                xaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                yaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if abs(correlation) > 0.7:
                st.success(f"Strong correlation detected ({correlation:.3f})")
            elif abs(correlation) > 0.4:
                st.warning(f"Moderate correlation ({correlation:.3f})")
            else:
                st.info(f"Weak correlation ({correlation:.3f})")
    
    elif viz_type == "Bar Chart":
        if cat_cols and numeric_cols:
            cat_col = st.selectbox("Category:", cat_cols, key="bar_category")
            num_col = st.selectbox("Value:", numeric_cols, key="bar_value")
            
            default_agg = "mean"
            if "count" in num_col.lower() or "total" in num_col.lower():
                default_agg = "sum"
            
            agg_options = ["mean", "median", "sum", "count"]
            default_idx = agg_options.index(default_agg)
            agg_func = st.selectbox("Aggregation:", agg_options, index=default_idx, key="bar_agg")
            
            grouped = df.groupby(cat_col)[num_col].agg(agg_func).reset_index()
            
            if len(grouped) > 30:
                st.info(f"Showing top 30 of {len(grouped)} categories")
                grouped = grouped.sort_values(num_col, ascending=False).head(30)
            
            fig = px.bar(
                grouped, x=cat_col, y=num_col,
                title=f"{agg_func.title()} of {num_col} by {cat_col}",
                color_discrete_sequence=['#06b6d4']
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                xaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                yaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        if numeric_cols:
            st.info("Box plots show the distribution, median, quartiles, and outliers.")
            col = st.selectbox("Select column:", numeric_cols, key="box_column")
            group_by = st.selectbox("Group by (optional):", ["None"] + cat_cols[:10], key="box_group")
            
            plot_df = df if len(df) <= 10000 else df.sample(10000)
            
            if group_by == "None":
                fig = px.box(plot_df, y=col, title=f"Box Plot of {col}")
            else:
                top_groups = df[group_by].value_counts().head(15).index
                plot_df = plot_df[plot_df[group_by].isin(top_groups)]
                fig = px.box(plot_df, x=group_by, y=col, title=f"Box Plot of {col} by {group_by}")
            
            fig.update_traces(marker_color='#06b6d4', line_color='#0891b2')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                xaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                yaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap":
        if len(numeric_cols) >= 2:
            st.info("Darker colors indicate stronger correlations. Look for patterns and relationships.")
            
            max_corr_cols = 15
            if len(numeric_cols) > max_corr_cols:
                st.warning(f"Showing top {max_corr_cols} most variable columns for clarity.")
                variances = df[numeric_cols].var().sort_values(ascending=False)
                selected_cols = variances.head(max_corr_cols).index.tolist()
            else:
                selected_cols = numeric_cols
            
            corr_df = df[selected_cols] if len(df) <= 20000 else df[selected_cols].sample(20000)
            corr_matrix = corr_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='Teal',
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10, "color": "#e5e7eb"},
                zmin=-1, zmax=1
            ))
            
            fig.update_layout(
                title="Correlation Heatmap",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Line Chart":
        if numeric_cols:
            st.info("Line charts work best for time series or sequential data.")
            y_col = st.selectbox("Y-axis:", numeric_cols, key="line_y")
            x_col = st.selectbox("X-axis:", df.columns.tolist()[:20], key="line_x")
            
            max_points = 1000
            if len(df) > max_points:
                st.info(f"Sampling {max_points} points for performance.")
                plot_df = df.sample(max_points).sort_values(x_col)
            else:
                plot_df = df.sort_values(x_col)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_df[x_col],
                y=plot_df[y_col],
                mode='lines+markers',
                line=dict(color='#06b6d4', width=3),
                marker=dict(size=6, color='#22d3ee'),
                name=y_col
            ))
            
            fig.update_layout(
                title=f"{y_col} over {x_col}",
                xaxis_title=x_col,
                yaxis_title=y_col,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                xaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                yaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)


def export_features():
    """Export data and reports - ALL FREE"""
    st.subheader("Export Options")
    
    if st.session_state.df_filtered is None:
        return
    
    show_free_pro_banner()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        csv = st.session_state.df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_csv_button"
        )
    
    with col2:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            st.session_state.df_filtered.to_excel(writer, index=False, sheet_name='Data')
        
        st.download_button(
            label="📊 Download Excel",
            data=buffer.getvalue(),
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_button"
        )
    
    with col3:
        if st.session_state.chat_history:
            chat_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_history])
            st.download_button(
                label="💬 Download Chat",
                data=chat_text,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_chat_button"
            )
    
    with col4:
        # Reports are now FREE
        if st.button("📄 Generate Report", key="generate_report_button"):
            with st.spinner("Generating executive summary..."):
                from src.utils.report_generator import ReportGenerator
                
                quality_analyzer = DataQualityAnalyzer(st.session_state.df_filtered)
                quality_report = quality_analyzer.calculate_trust_score()
                
                report_gen = ReportGenerator(
                    st.session_state.df_filtered,
                    quality_report,
                    st.session_state.data_summary
                )
                
                st.session_state.narrative_report = report_gen.generate_markdown_report()
                st.session_state.html_report = report_gen.generate_html_report()
                st.success("Report generated!")
    
    if st.session_state.narrative_report:
        st.markdown("---")
        st.subheader("Narrative Report Downloads")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.download_button(
                label="Download Markdown",
                data=st.session_state.narrative_report,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="download_md_report"
            )
        
        with col_b:
            st.download_button(
                label="Download HTML",
                data=st.session_state.html_report,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="download_html_report"
            )
        
        with col_c:
            st.download_button(
                label="Download Text",
                data=st.session_state.narrative_report,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_txt_report"
            )
        
        with st.expander("Preview Report"):
            st.markdown(st.session_state.narrative_report)


def display_data_overview():
    """Enhanced data overview with visualizations and trust score - ALL FREE"""
    if st.session_state.df_filtered is None:
        return
    
    st.header("Data Overview")
    
    # Show free banner if sales data detected
    if st.session_state.is_sales_data:
        st.info("💰 Sales data detected! Check out the **Sales Pipeline** tab for specialized analysis.")
    
    df = st.session_state.df_filtered
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("Memory", f"{memory_mb:.2f} MB")
    with col4:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    
    # Trust Score is now FREE
    st.markdown("---")
    st.subheader("Data Quality & Trust Score")
    
    show_free_pro_banner()
    
    with st.spinner("Analyzing data quality..."):
        quality_analyzer = DataQualityAnalyzer(df)
        quality_report = quality_analyzer.calculate_trust_score()
    
    score_col1, score_col2 = st.columns([1, 2])
    
    with score_col1:
        score = quality_report['overall_score']
        grade = quality_report['grade']
        
        if score >= 80:
            color = "#10b981"
        elif score >= 60:
            color = "#f59e0b"
        else:
            color = "#ef4444"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%); border-radius: 12px; border: 1px solid rgba(6, 182, 212, 0.3);'>
            <h1 style='color: {color}; font-size: 3rem; margin: 0;'>{score}</h1>
            <p style='font-size: 1.2rem; color: #9ca3af; margin: 5px 0;'>Trust Score</p>
            <p style='font-size: 1rem; color: #6b7280;'>{grade}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with score_col2:
        st.write("**Component Scores:**")
        for component, comp_score in quality_report['component_scores'].items():
            st.progress(comp_score / 100, text=f"{component.title()}: {comp_score:.0f}/100")
    
    col_issues, col_recommendations = st.columns(2)
    
    with col_issues:
        if quality_report['issues']:
            st.error("**Critical Issues:**")
            for issue in quality_report['issues']:
                st.write(f"- {issue}")
        
        if quality_report['warnings']:
            st.warning("**Warnings:**")
            for warning in quality_report['warnings']:
                st.write(f"- {warning}")
    
    with col_recommendations:
        st.info("**Recommendations:**")
        for rec in quality_report['recommendations']:
            st.write(f"- {rec}")
    
    st.markdown("---")
    
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    create_visualizations(df)
    
    export_features()


def display_statistics():
    """Display statistical analysis"""
    if st.session_state.df_filtered is None:
        return
    
    st.header("Statistical Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Numeric Stats", "Categorical Stats", "Correlations", "Advanced"])
    
    with tab1:
        numeric_stats = DataAnalyzer.get_numeric_statistics(st.session_state.df_filtered)
        if numeric_stats:
            st.dataframe(pd.DataFrame(numeric_stats).T, use_container_width=True)
        else:
            st.info("No numeric columns found")
    
    with tab2:
        categorical_stats = DataAnalyzer.get_categorical_statistics(st.session_state.df_filtered)
        if categorical_stats:
            for col, stats in categorical_stats.items():
                with st.expander(f"{col}"):
                    st.write(f"Unique values: {stats['unique_count']}")
                    st.write(f"Missing values: {stats['null_count']}")
                    st.write(f"Most common: {stats['most_common']}")
                    if stats['value_counts']:
                        st.bar_chart(stats['value_counts'])
        else:
            st.info("No categorical columns found")
    
    with tab3:
        correlations = DataAnalyzer.find_correlations(st.session_state.df_filtered, threshold=0.5)
        if correlations:
            st.dataframe(pd.DataFrame(correlations), use_container_width=True)
        else:
            st.info("No significant correlations found")
    
    with tab4:
        clean_data_interface()


def forecasting_interface():
    """Time-series forecasting interface - NOW FREE"""
    st.header("⏰ Time-Series Forecasting")
    
    show_free_pro_banner()
    
    if st.session_state.df_filtered is None:
        st.info("Please load data first")
        return
    
    df = st.session_state.df_filtered
    
    quality_analyzer = DataQualityAnalyzer(df)
    quality_report = quality_analyzer.calculate_trust_score()
    
    score = quality_report['overall_score']
    
    if score < 100:
        st.warning(f"Data Trust Score: {score}/100. Forecasting works best with complete, clean data.")
    else:
        st.success(f"Data Trust Score: {score}/100. Excellent data quality!")
    
    st.markdown("---")
    
    from src.utils.forecaster import TimeSeriesForecaster
    forecaster = TimeSeriesForecaster(df)
    
    time_cols = forecaster.detect_time_columns()
    
    if not time_cols:
        st.error("No date/time columns detected in the dataset.")
        return
    
    st.subheader("Configure Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_col = st.selectbox(
            "Select Date Column:",
            time_cols,
            key="forecast_date_col"
        )
    
    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found for forecasting.")
            return
        
        value_col = st.selectbox(
            "Select Value to Forecast:",
            numeric_cols,
            key="forecast_value_col"
        )
    
    method = st.radio(
        "Forecasting Method:",
        ["Exponential Smoothing (Recommended)", "ARIMA"],
        key="forecast_method"
    )
    
    if st.button("Generate 6-Month Forecast", key="generate_forecast_button"):
        with st.spinner("Preparing time series data..."):
            try:
                series, validation = forecaster.prepare_time_series(date_col, value_col)
                
                st.subheader("Data Validation")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Total Data Points", validation['total_points'])
                with col_b:
                    st.metric("Frequency", validation['frequency'])
                with col_c:
                    st.metric("Date Range", f"{validation['date_range'][0].strftime('%Y-%m-%d')} to {validation['date_range'][1].strftime('%Y-%m-%d')}")
                
                if validation['total_points'] < 24:
                    st.error("Insufficient data for forecasting.")
                    return
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error preparing data: {str(e)}")
                return
        
        with st.spinner("Generating forecast..."):
            try:
                if "Exponential" in method:
                    fitted, forecast_df = forecaster.forecast_exponential_smoothing(series, periods=180)
                else:
                    fitted, forecast_df = forecaster.forecast_arima(series, periods=180)
                
                metrics = forecaster.evaluate_forecast(series, fitted)
                
                st.subheader("Forecast Accuracy Metrics")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                with metric_col2:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with metric_col3:
                    st.metric("MAPE", f"{metrics['mape']:.1f}%")
                with metric_col4:
                    st.metric("R²", f"{metrics['r_squared']:.3f}")
                
                st.markdown("---")
                
                st.subheader("Forecast Visualization")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='#06b6d4', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=fitted.index,
                    y=fitted.values,
                    mode='lines',
                    name='Fitted Values',
                    line=dict(color='#f59e0b', dash='dash', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['forecast'].values,
                    mode='lines',
                    name='6-Month Forecast',
                    line=dict(color='#ef4444', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['upper_bound'].values,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['lower_bound'].values,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(239, 68, 68, 0.2)',
                    name='95% Confidence'
                ))
                
                fig.update_layout(
                    title=f"{value_col} - Historical & 6-Month Forecast",
                    xaxis_title="Date",
                    yaxis_title=value_col,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e5e7eb'),
                    xaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                    yaxis=dict(gridcolor='rgba(55, 65, 81, 0.3)'),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Forecast Summary")
                
                last_actual = series.iloc[-1]
                last_forecast = forecast_df['forecast'].iloc[-1]
                percent_change = ((last_forecast - last_actual) / last_actual) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Value", f"{last_actual:.2f}")
                with col2:
                    st.metric("6-Month Forecast", f"{last_forecast:.2f}")
                with col3:
                    st.metric("Expected Change", f"{percent_change:+.1f}%")
                
                st.markdown("---")
                st.subheader("Export Forecast")
                
                export_df = pd.DataFrame({
                    'date': forecast_df.index,
                    'forecast': forecast_df['forecast'].values,
                    'lower_95': forecast_df['lower_bound'].values,
                    'upper_95': forecast_df['upper_bound'].values
                })
                
                csv_export = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv_export,
                    file_name=f"forecast_{value_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                return


def scenario_simulation_interface():
    """What-if scenario simulation interface - NOW FREE"""
    st.header("🎯 Scenario Simulation")
    
    show_free_pro_banner()
    
    if st.session_state.df_filtered is None:
        st.info("Please load data first")
        return
    
    df = st.session_state.df_filtered
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Scenario simulation requires at least 2 numeric variables.")
        return
    
    st.info("Simulate business scenarios by changing variables and see predicted impacts.")
    
    from src.utils.scenario_simulator import ScenarioSimulator
    simulator = ScenarioSimulator(df)
    
    st.markdown("---")
    
    with st.expander("View Variable Correlations"):
        st.write("**Strong correlations detected:**")
        correlations = simulator.correlations
        
        corr_pairs = []
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                if abs(correlations.iloc[i, j]) >= 0.3:
                    corr_pairs.append((
                        correlations.columns[i],
                        correlations.columns[j],
                        correlations.iloc[i, j]
                    ))
        
        if corr_pairs:
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for var1, var2, corr in corr_pairs[:10]:
                st.write(f"- **{var1}** ↔ **{var2}**: {corr:.3f}")
        else:
            st.write("No strong correlations found")
    
    st.markdown("---")
    
    st.subheader("Configure Scenario")
    
    input_method = st.radio(
        "Input Method:",
        ["Text Description", "Manual Selection"],
        key="scenario_input_method"
    )
    
    changes = []
    
    if input_method == "Text Description":
        st.write("**Enter your scenario in plain English:**")
        st.caption("Examples: 'increase revenue by 20%' or 'increase sales 15% and reduce costs 10%'")
        
        scenario_text = st.text_input(
            "Scenario:",
            placeholder="increase revenue by 20%",
            key="scenario_text_input"
        )
        
        if scenario_text:
            changes = simulator.parse_scenario_text(scenario_text)
            
            if changes:
                st.success(f"Parsed {len(changes)} change(s)")
                for change in changes:
                    st.write(f"- {change['action'].title()} **{change['variable']}** by {abs(change['change_percent']):.1f}%")
    
    else:
        num_changes = st.slider("Number of variables:", 1, 3, 1, key="num_changes")
        
        for i in range(num_changes):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                variable = st.selectbox(
                    f"Variable {i+1}:",
                    numeric_cols,
                    key=f"var_select_{i}"
                )
            
            with col2:
                change_pct = st.number_input(
                    f"Change %:",
                    value=10.0,
                    step=5.0,
                    key=f"change_pct_{i}"
                )
            
            changes.append({
                'variable': variable,
                'change_percent': change_pct
            })
    
    corr_threshold = st.slider(
        "Correlation threshold:",
        0.1, 0.9, 0.3, 0.1,
        key="corr_threshold"
    )
    
    if st.button("Run Simulation", key="run_simulation_button") and changes:
        with st.spinner("Simulating scenario..."):
            results = simulator.simulate_scenario(changes, corr_threshold)
            
            st.markdown("---")
            st.subheader("Simulation Results")
            
            st.write("### Primary Changes")
            for change in results['primary_changes']:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        change['variable'],
                        f"{change['new_value']:.2f}",
                        f"{change['change_percent']:+.1f}%"
                    )
                
                with col2:
                    st.write(f"**Current:** {change['current_value']:.2f}")
                    st.write(f"**New:** {change['new_value']:.2f}")
                
                with col3:
                    st.write(f"**Change:**")
                    st.write(f"{change['absolute_change']:+.2f}")
            
            st.markdown("---")
            
            if results['secondary_effects']:
                st.write("### Predicted Secondary Effects")
                
                for effect in results['secondary_effects']:
                    with st.expander(f"{effect['variable']} (corr: {effect['correlation']:.3f})"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric(
                                "Estimated New Value",
                                f"{effect['estimated_value']:.2f}",
                                f"{effect['estimated_change_percent']:+.1f}%"
                            )
                        
                        with col_b:
                            st.write(f"**Current:** {effect['current_value']:.2f}")
                            st.write(f"**Estimated:** {effect['estimated_value']:.2f}")
            else:
                st.info("No significant secondary effects detected.")


def chat_interface():
    """AI chat interface - NOW UNLIMITED AND FREE"""
    st.header("💬 Chat with Your Data")
    
    show_free_pro_banner()
    
    if st.session_state.df_filtered is None:
        st.info("Please load data first")
        return
    
    if st.session_state.ai_agent is None:
        st.warning("AI Chat is not available. Please check your API configuration.")
        return
    
    st.success("✨ Unlimited AI questions - Free for testing!")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.ai_agent.analyze_data(
                        st.session_state.df_filtered,
                        prompt,
                        st.session_state.data_summary
                    )
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)


def subscription_interface():
    """Display subscription management interface"""
    st.subheader("💳 Subscription Management")
    
    show_free_pro_banner()
    
    username = st.session_state.get('username', '')
    
    if not username:
        st.warning("Please log in to manage subscriptions")
        return
    
    sub_manager = SubscriptionManager()
    stripe_handler = StripeHandler()
    
    subscription = sub_manager.get_user_subscription(username)
    current_tier = subscription.get('tier', 'free')
    
    st.markdown(f"**Current Plan:** {sub_manager.TIERS[current_tier]['name']}")
    
    st.markdown("---")
    
    col_free, col_pro = st.columns(2)
    
    with col_free:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%); padding: 2rem; border-radius: 12px; border: 1px solid rgba(6, 182, 212, 0.3);'>
            <h3 style='color: #22d3ee; margin-top: 0;'>Free</h3>
            <h2 style='color: #e5e7eb; margin: 0.5rem 0;'>$0/month</h2>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        for feature in sub_manager.TIERS['free']['features']:
            st.write(f"✓ {feature}")
        st.write("")
        if current_tier == 'free':
            st.success("✓ Current Plan")
    
    with col_pro:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.2) 100%); padding: 2rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.4);'>
            <h3 style='color: #f59e0b; margin-top: 0;'>Pro <span style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem;'>Popular</span></h3>
            <h2 style='color: #e5e7eb; margin: 0.5rem 0;'>$24.99/month</h2>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        for feature in sub_manager.TIERS['pro']['features']:
            st.write(f"✓ {feature}")
        st.write("✓ Sales Pipeline Analyzer")
        st.write("")
        
        st.info("🎉 All Pro features are currently FREE for testing!")
        
        if current_tier == 'free':
            if stripe_handler.is_configured():
                user_email = st.session_state.get('user_info', {}).get('email', '')
                
                if not user_email:
                    user_email = st.text_input("Email for billing:", key="billing_email")
                
                if user_email and st.button("Subscribe Now (Coming Soon)", key="subscribe_pro_button", type="primary", disabled=True):
                    st.info("Payment processing will be enabled after testing period")
            else:
                st.info("Test all Pro features now - No payment required!")
        else:
            st.success("✓ Current Plan")
    
    st.markdown("---")
    st.subheader("Feature Access")
    
    st.success("✅ All features are currently FREE for testing!")
    
    features_list = [
        'Time-Series Forecasting',
        'Scenario Simulation',
        'Data Quality Trust Score',
        'Narrative Reports',
        'Unlimited AI Chat',
        'Sales Pipeline Analyzer'
    ]
    
    for feature in features_list:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.write(feature)
        with col_b:
            st.success("Free")


def main():
    apply_custom_styles()
    initialize_session_state()
    
    if not st.session_state.authenticated:
        login_page()
        return
    
    user_name = st.session_state.get('user_info', {}).get("name", st.session_state.get('username', 'User'))
    st.markdown(f'<p class="user-info">Logged in as: <strong>{user_name}</strong></p>', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">AI DataChat</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Universal Intelligence Through Data</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("📁 Upload Data")
        
        if st.button("🚪 Logout", key="logout_button", use_container_width=True):
            logout()
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload CSV, Excel, or JSON files (max 2GB)"
        )
        
        if uploaded_file:
            if st.button("📂 Load Data", key="load_data_button", use_container_width=True):
                with st.spinner("Loading data..."):
                    if load_data_file(uploaded_file):
                        st.success("✅ Data loaded successfully!")
                        if st.session_state.is_sales_data:
                            st.info("💰 Sales data detected!")
        
        if st.session_state.df is not None:
            st.markdown("---")
            st.subheader("📊 File Info")
            st.write(f"**Name:** {st.session_state.file_info['name']}")
            st.write(f"**Format:** {st.session_state.file_info['format']}")
            st.write(f"**Size:** {st.session_state.file_info['size_mb']} MB")
            st.write(f"**Rows:** {len(st.session_state.df):,}")
            
            if st.session_state.is_sales_data:
                st.markdown("---")
                st.success("💰 Sales Data Detected!")
                st.caption("Check the Sales Pipeline tab")
            
            st.markdown("---")
            data_filtering_interface()
            
            if st.button("🔄 Reset Filters", key="reset_filters_button", use_container_width=True):
                st.session_state.df_filtered = st.session_state.df.copy()
                st.rerun()
            
            if st.button("🗑️ Clear Data", key="clear_data_button", use_container_width=True):
                for key in ['df', 'df_filtered', 'file_info', 'data_summary', 'ai_agent', 'chat_history', 'narrative_report', 'html_report', 'is_sales_data']:
                    st.session_state[key] = None if key != 'chat_history' else []
                st.session_state.is_sales_data = False
                st.rerun()
    
    if st.session_state.df is None:
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem;'>
            <h1 style='font-size: 4rem; margin-bottom: 1rem;'>📊</h1>
            <h2 style='color: #22d3ee; font-weight: 300; margin-bottom: 1rem;'>Welcome to AI DataChat</h2>
            <p style='color: #9ca3af; font-size: 1.2rem; margin-bottom: 2rem;'>
                Transform your data into actionable insights with AI-powered analytics
            </p>
            <p style='color: #6b7280;'>
                ← Start by uploading a file in the sidebar
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        show_free_pro_banner()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%); padding: 2rem; border-radius: 12px; border: 1px solid rgba(6, 182, 212, 0.3);'>
                <h3 style='color: #22d3ee;'>✨ What you can do:</h3>
                <ul style='color: #d1d5db;'>
                    <li>Upload and analyze CSV, Excel, JSON files</li>
                    <li>Interactive visualizations and statistics</li>
                    <li>Data cleaning and filtering</li>
                    <li>AI-powered chat with your data</li>
                    <li>Sales Pipeline Analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%); padding: 2rem; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.4);'>
                <h3 style='color: #10b981;'>🎉 All Features FREE:</h3>
                <ul style='color: #d1d5db;'>
                    <li>Time-series forecasting (6-month predictions)</li>
                    <li>Scenario simulation (what-if analysis)</li>
                    <li>Data quality trust scores</li>
                    <li>Automated narrative reports</li>
                    <li>Unlimited AI questions</li>
                    <li>Sales Pipeline Analyzer</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Dynamic tabs based on sales data detection
        if st.session_state.is_sales_data:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Overview", 
                "📈 Statistics", 
                "💰 Sales Pipeline",
                "⏰ Forecasting", 
                "🎯 Scenarios", 
                "💬 Chat", 
                "💳 Subscription"
            ])
            
            with tab3:
                sales_pipeline_analyzer()
        else:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview", 
                "📈 Statistics", 
                "⏰ Forecasting", 
                "🎯 Scenarios", 
                "💬 Chat", 
                "💳 Subscription"
            ])
        
        with tab1:
            display_data_overview()
        
        with tab2:
            display_statistics()
        
        with tab3 if not st.session_state.is_sales_data else tab4:
            forecasting_interface()
        
        with tab4 if not st.session_state.is_sales_data else tab5:
            scenario_simulation_interface()
        
        with tab5 if not st.session_state.is_sales_data else tab6:
            chat_interface()
        
        with tab6 if not st.session_state.is_sales_data else tab7:
            subscription_interface()


if __name__ == "__main__":
    main()
