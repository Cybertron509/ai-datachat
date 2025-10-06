"""
AI DataChat - Complete Application with Subscriptions and Monetization
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

from src.utils.file_handler import FileHandler
from src.utils.data_analyzer import DataAnalyzer
from src.agents.ai_agent import AIAgent
from src.utils.auth import AuthManager
from src.utils.data_quality import DataQualityAnalyzer
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def login_page():
    """Display login page"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .tagline {
            font-size: 1.2rem;
            color: #888;
            text-align: center;
            margin-bottom: 2rem;
            font-style: italic;
        }
    </style>
    """, unsafe_allow_html=True)
    
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
    st.rerun()


def load_data_file(uploaded_file):
    """Load and process uploaded file"""
    try:
        temp_path = Path(f"temp_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = FileHandler.load_file(temp_path)
        file_info = FileHandler.get_file_info(temp_path)
        temp_path.unlink()
        
        data_summary = DataAnalyzer.get_summary_statistics(df)
        
        st.session_state.df = df
        st.session_state.df_filtered = df.copy()
        st.session_state.file_info = file_info
        st.session_state.data_summary = data_summary
        
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


def clean_data_interface():
    """Data cleaning interface"""
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
            df = st.session_state.df_filtered.copy()
            
            if missing_action == "Drop rows with any missing":
                df = df.dropna()
            elif missing_action == "Drop rows with all missing":
                df = df.dropna(how='all')
            elif missing_action == "Fill with mean":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif missing_action == "Fill with median":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif missing_action == "Fill with mode":
                for col in df.columns:
                    if not df[col].mode().empty:
                        df[col] = df[col].fillna(df[col].mode()[0])
            
            st.session_state.df_filtered = df
            st.success(f"Applied: {missing_action}")
            st.rerun()
    
    with col2:
        st.write("**Handle Outliers**")
        numeric_cols = st.session_state.df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            outlier_col = st.selectbox("Select column:", numeric_cols, key="outlier_column_selector")
            outlier_method = st.selectbox("Method:", ["IQR", "Z-Score"], key="outlier_method_selector")
            
            if st.button("Remove Outliers", key="remove_outliers_button"):
                df = st.session_state.df_filtered.copy()
                
                if outlier_method == "IQR":
                    Q1 = df[outlier_col].quantile(0.25)
                    Q3 = df[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[(df[outlier_col] >= Q1 - 1.5 * IQR) & (df[outlier_col] <= Q3 + 1.5 * IQR)]
                else:
                    z_scores = np.abs((df[outlier_col] - df[outlier_col].mean()) / df[outlier_col].std())
                    df = df[z_scores < 3]
                
                st.session_state.df_filtered = df
                st.success(f"Removed outliers from {outlier_col}")
                st.rerun()


def data_filtering_interface():
    """Advanced data filtering"""
    st.subheader("Filter Data")
    
    if st.session_state.df is None:
        return
    
    df = st.session_state.df.copy()
    
    # Column selection
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to display:", 
        all_columns, 
        default=all_columns,
        key="column_filter_selector"
    )
    
    # Numeric filters
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
    
    # Categorical filters
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
    """Create interactive visualizations"""
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
            
            fig = px.histogram(plot_df, x=col, title=f"Distribution of {col}", nbins=50)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Mean:** {df[col].mean():.2f} | **Median:** {df[col].median():.2f} | **Std:** {df[col].std():.2f}")
    
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
            
            fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, 
                           title=f"{x_col} vs {y_col} (Correlation: {correlation:.3f})",
                           opacity=0.6)
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
            
            if agg_func == "sum" and num_col.lower() in ["age", "bmi", "rating", "score", "hours", "sleep_hours", "sleep"]:
                st.warning(f"Note: Summing '{num_col}' may not be meaningful. Consider using 'mean' or 'median' instead.")
            
            grouped = df.groupby(cat_col)[num_col].agg(agg_func).reset_index()
            
            if len(grouped) > 30:
                st.info(f"Showing top 30 of {len(grouped)} categories")
                grouped = grouped.sort_values(num_col, ascending=False).head(30)
            
            fig = px.bar(grouped, x=cat_col, y=num_col, 
                        title=f"{agg_func.title()} of {num_col} by {cat_col}")
            st.plotly_chart(fig, use_container_width=True)
            
            if agg_func in ["mean", "median"]:
                max_cat = grouped.iloc[0][cat_col]
                max_val = grouped.iloc[0][num_col]
                st.info(f"Highest {agg_func}: {max_cat} ({max_val:.2f})")
    
    elif viz_type == "Box Plot":
        if numeric_cols:
            st.info("Box plots show the distribution, median, quartiles, and outliers.")
            col = st.selectbox("Select column:", numeric_cols, key="box_column")
            group_by = st.selectbox("Group by (optional):", ["None"] + cat_cols[:10], key="box_group")
            
            plot_df = df if len(df) <= 10000 else df.sample(10000)
            if len(df) > 10000:
                st.caption(f"Showing sample of 10,000 from {len(df):,} total rows")
            
            if group_by == "None":
                fig = px.box(plot_df, y=col, title=f"Box Plot of {col}")
            else:
                top_groups = df[group_by].value_counts().head(15).index
                plot_df = plot_df[plot_df[group_by].isin(top_groups)]
                fig = px.box(plot_df, x=group_by, y=col, title=f"Box Plot of {col} by {group_by}")
            
            st.plotly_chart(fig, use_container_width=True)
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                st.warning(f"Detected {len(outliers):,} outliers ({len(outliers)/len(df)*100:.1f}% of data)")
    
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
            
            fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", 
                          title="Correlation Heatmap",
                          color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
            
            corr_flat = corr_matrix.abs().unstack()
            corr_flat = corr_flat[corr_flat < 1].sort_values(ascending=False).head(3)
            if len(corr_flat) > 0:
                st.write("**Strongest correlations:**")
                for (var1, var2), corr_val in corr_flat.items():
                    actual_corr = corr_matrix.loc[var1, var2]
                    st.write(f"- {var1} & {var2}: {actual_corr:.3f}")
    
    elif viz_type == "Line Chart":
        if numeric_cols:
            st.info("Line charts work best for time series or sequential data.")
            y_col = st.selectbox("Y-axis:", numeric_cols, key="line_y")
            x_col = st.selectbox("X-axis:", df.columns.tolist()[:20], key="line_x")
            
            if df[x_col].dtype == 'object':
                st.warning(f"'{x_col}' is categorical. Line charts work best with sequential/time data.")
            
            max_points = 1000
            if len(df) > max_points:
                st.info(f"Sampling {max_points} points for performance.")
                plot_df = df.sample(max_points).sort_values(x_col)
            else:
                plot_df = df.sort_values(x_col)
            
            fig = px.line(plot_df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
            st.plotly_chart(fig, use_container_width=True)


def export_features():
    """Export data and reports"""
    st.subheader("Export Options")
    
    if st.session_state.df_filtered is None:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        csv = st.session_state.df_filtered.to_csv(index=False)
        st.download_button(
            label="Download CSV",
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
            label="Download Excel",
            data=buffer.getvalue(),
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_button"
        )
    
    with col3:
        if st.session_state.chat_history:
            chat_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_history])
            st.download_button(
                label="Download Chat",
                data=chat_text,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_chat_button"
            )
    
    with col4:
        from src.utils.subscription import SubscriptionManager
        sub_manager = SubscriptionManager()
        
        if not sub_manager.can_access_feature(st.session_state.username, 'reports'):
            st.info("Report generation is a Pro feature")
            if st.button("Upgrade", key="upgrade_reports_btn"):
                st.session_state.show_pricing = True
                st.rerun()
            return
        
        if st.button("Generate Report", key="generate_report_button"):
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
                st.success("Report generated! Download options below.")
    
    if st.session_state.narrative_report:
        st.markdown("---")
        st.subheader("Narrative Report Downloads")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.download_button(
                label="Download as Markdown",
                data=st.session_state.narrative_report,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="download_md_report"
            )
        
        with col_b:
            st.download_button(
                label="Download as HTML",
                data=st.session_state.html_report,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="download_html_report"
            )
        
        with col_c:
            st.download_button(
                label="Download as Text",
                data=st.session_state.narrative_report,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_txt_report"
            )
        
        with st.expander("Preview Report"):
            st.markdown(st.session_state.narrative_report)


def display_data_overview():
    """Enhanced data overview with visualizations and trust score"""
    if st.session_state.df_filtered is None:
        return
    
    st.header("Data Overview")
    
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
    
    from src.utils.subscription import SubscriptionManager
    sub_manager = SubscriptionManager()
    
    if not sub_manager.can_access_feature(st.session_state.username, 'trust_score'):
        st.markdown("---")
        st.info("Data Quality Trust Score is a Pro feature")
        st.write("Upgrade to unlock comprehensive data quality analysis with actionable recommendations.")
        if st.button("View Pricing", key="upgrade_trust_score"):
            st.session_state.show_pricing = True
            st.rerun()
    else:
        st.markdown("---")
        st.subheader("Data Quality & Trust Score")
        
        with st.spinner("Analyzing data quality..."):
            quality_analyzer = DataQualityAnalyzer(df)
            quality_report = quality_analyzer.calculate_trust_score()
        
        score_col1, score_col2 = st.columns([1, 2])
        
        with score_col1:
            score = quality_report['overall_score']
            grade = quality_report['grade']
            
            if score >= 80:
                color = "green"
            elif score >= 60:
                color = "orange"
            else:
                color = "red"
            
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h1 style='color: {color}; font-size: 3rem; margin: 0;'>{score}</h1>
                <p style='font-size: 1.2rem; color: #666; margin: 5px 0;'>Trust Score</p>
                <p style='font-size: 1rem; color: #888;'>{grade}</p>
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
            
            if quality_report['info']:
                with st.expander("Additional Information"):
                    for info in quality_report['info']:
                        st.write(f"- {info}")
        
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
    """Time-series forecasting interface"""
    
    from src.utils.subscription import SubscriptionManager
    sub_manager = SubscriptionManager()
    
    if not sub_manager.can_access_feature(st.session_state.username, 'forecasting'):
        st.header("Time-Series Forecasting")
        st.warning("Time-Series Forecasting is a Pro feature")
        st.info("Upgrade to Pro to unlock 6-month predictions with confidence intervals, accuracy metrics, and CSV export.")
        if st.button("View Pricing", key="upgrade_forecasting"):
            st.session_state.show_pricing = True
            st.rerun()
        return
    
    st.header("Time-Series Forecasting")
    
    if st.session_state.df_filtered is None:
        st.info("Please load data first")
        return
    
    df = st.session_state.df_filtered
    
    quality_analyzer = DataQualityAnalyzer(df)
    quality_report = quality_analyzer.calculate_trust_score()
    
    score = quality_report['overall_score']
    
    if score < 100:
        st.warning(f"Data Trust Score: {score}/100. Forecasting works best with complete, clean data (100% trust score). Missing or inconsistent data may produce less accurate predictions.")
    else:
        st.success(f"Data Trust Score: {score}/100. Excellent data quality for forecasting!")
    
    st.markdown("---")
    
    from src.utils.forecaster import TimeSeriesForecaster
    forecaster = TimeSeriesForecaster(df)
    
    time_cols = forecaster.detect_time_columns()
    
    if not time_cols:
        st.error("No date/time columns detected in the dataset. Forecasting requires a date column.")
        st.info("**Tip:** Ensure your dataset has a column with dates in a recognizable format (e.g., '2024-01-01', '01/01/2024', etc.)")
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
                    st.error("Insufficient data for forecasting. At least 24 data points (2 years of monthly data) are recommended for reliable predictions.")
                    return
                
                if validation['missing_count'] > 0:
                    st.warning(f"Found {validation['missing_count']} missing values. These have been removed, which may affect forecast accuracy.")
                
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
                
                st.caption("MAE: Mean Absolute Error | RMSE: Root Mean Square Error | MAPE: Mean Absolute Percentage Error | R²: Coefficient of Determination")
                
                st.markdown("---")
                
                st.subheader("Forecast Visualization")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=fitted.index,
                    y=fitted.values,
                    mode='lines',
                    name='Fitted Values',
                    line=dict(color='orange', dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['forecast'].values,
                    mode='lines',
                    name='6-Month Forecast',
                    line=dict(color='red')
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
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    name='95% Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f"{value_col} - Historical Data and 6-Month Forecast",
                    xaxis_title="Date",
                    yaxis_title=value_col,
                    hovermode='x unified',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Forecast Summary")
                
                last_actual = series.iloc[-1]
                last_forecast = forecast_df['forecast'].iloc[-1]
                percent_change = ((last_forecast - last_actual) / last_actual) * 100
                
                st.write(f"**Current Value ({series.index[-1].strftime('%Y-%m-%d')}):** {last_actual:.2f}")
                st.write(f"**Forecasted Value (6 months ahead):** {last_forecast:.2f}")
                st.write(f"**Expected Change:** {percent_change:+.1f}%")
                
                if percent_change > 0:
                    st.success(f"The model predicts an increase of {abs(percent_change):.1f}% over the next 6 months.")
                else:
                    st.warning(f"The model predicts a decrease of {abs(percent_change):.1f}% over the next 6 months.")
                
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
                    label="Download Forecast as CSV",
                    data=csv_export,
                    file_name=f"forecast_{value_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                st.info("Try using a different forecasting method or check your data format.")
                return


def scenario_simulation_interface():
    """What-if scenario simulation interface"""
    
    from src.utils.subscription import SubscriptionManager
    sub_manager = SubscriptionManager()
    
    if not sub_manager.can_access_feature(st.session_state.username, 'scenarios'):
        st.header("Scenario Simulation - What If Analysis")
        st.warning("Scenario Simulation is a Pro feature")
        st.info("Upgrade to Pro to unlock multi-variable what-if analysis with sensitivity testing and correlation-based predictions.")
        if st.button("View Pricing", key="upgrade_scenarios"):
            st.session_state.show_pricing = True
            st.rerun()
        return
    
    st.header("Scenario Simulation - What If Analysis")
    
    if st.session_state.df_filtered is None:
        st.info("Please load data first")
        return
    
    df = st.session_state.df_filtered
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Scenario simulation requires at least 2 numeric variables to analyze relationships.")
        return
    
    st.info("Simulate business scenarios by changing variables and see the predicted impact on correlated metrics.")
    
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
            st.write("No strong correlations found (threshold: 0.3)")
    
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
                st.success(f"Parsed {len(changes)} change(s):")
                for change in changes:
                    st.write(f"- {change['action'].title()} **{change['variable']}** by {abs(change['change_percent']):.1f}%")
            else:
                st.warning("Could not parse scenario. Try manual selection or use format: 'increase [variable] by [number]%'")
    
    else:
        num_changes = st.slider("Number of variables to change:", 1, 3, 1, key="num_changes")
        
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
        "Correlation threshold (minimum to show secondary effects):",
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
                    st.write(f"**Absolute Change:**")
                    st.write(f"{change['absolute_change']:+.2f}")
            
            st.markdown("---")
            
            if results['secondary_effects']:
                st.write("### Predicted Secondary Effects")
                st.caption("Based on historical correlations between variables")
                
                for effect in results['secondary_effects']:
                    with st.expander(f"{effect['variable']} (corr: {effect['correlation']:.3f} with {effect['affected_by']})"):
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
                            st.write(f"**Change:** {effect['absolute_change']:+.2f}")
                        
                        confidence = "High" if abs(effect['correlation']) > 0.7 else "Medium" if abs(effect['correlation']) > 0.5 else "Low"
                        st.caption(f"Confidence: {confidence} (based on correlation strength)")
            else:
                st.info("No significant secondary effects detected at the current correlation threshold.")
            
            st.markdown("---")
            
            st.write("### Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Variables Changed Directly", results['summary']['total_variables_changed'])
            with summary_col2:
                st.metric("Predicted Secondary Effects", results['summary']['total_secondary_effects'])
            with summary_col3:
                st.metric("Total Variables Affected", results['summary']['total_variables_affected'])
    
    if changes and len(changes) == 1:
        st.markdown("---")
        st.subheader("Sensitivity Analysis")
        st.write("See how outcomes change with different percentage adjustments")
        
        if st.button("Run Sensitivity Analysis", key="run_sensitivity_button"):
            variable = changes[0]['variable']
            base_change = changes[0]['change_percent']
            
            with st.spinner("Running sensitivity analysis..."):
                sensitivity = simulator.sensitivity_analysis(variable, base_change, range_percent=10)
                
                st.write(f"**Testing {variable} changes from {base_change-10:.1f}% to {base_change+10:.1f}%**")
                
                sens_col1, sens_col2, sens_col3 = st.columns(3)
                
                with sens_col1:
                    st.metric(
                        "Pessimistic (-10%)",
                        f"{sensitivity['pessimistic_scenario']['new_value']:.2f}",
                        f"{sensitivity['pessimistic_scenario']['change_percent']:.1f}%"
                    )
                
                with sens_col2:
                    st.metric(
                        "Base Scenario",
                        f"{sensitivity['base_scenario']['new_value']:.2f}",
                        f"{sensitivity['base_scenario']['change_percent']:.1f}%"
                    )
                
                with sens_col3:
                    st.metric(
                        "Optimistic (+10%)",
                        f"{sensitivity['optimistic_scenario']['new_value']:.2f}",
                        f"{sensitivity['optimistic_scenario']['change_percent']:.1f}%"
                    )
                
                st.write(f"**Sensitivity Coefficient:** {sensitivity['sensitivity_coefficient']:.2f}")
                st.caption("This shows how much the outcome changes per 1% input change. Higher = more sensitive to changes.")
                
                scenarios_data = sensitivity['scenarios']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[s['change_percent'] for s in scenarios_data],
                    y=[s['new_value'] for s in scenarios_data],
                    mode='lines+markers',
                    name=variable,
                    line=dict(color='blue', width=3),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title=f"Sensitivity Analysis: {variable}",
                    xaxis_title="Change Percentage (%)",
                    yaxis_title="Resulting Value",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)


def chat_interface():
    """AI chat interface with per-user rate limiting"""
    st.header("Chat with Your Data")
    
    if st.session_state.df_filtered is None:
        st.info("Please load data first")
        return
    
    if st.session_state.ai_agent is None:
        st.warning("AI Chat is not available. Please check your OpenAI API configuration and ensure you have sufficient credits.")
        return
    
    from src.utils.subscription import SubscriptionManager
    
    try:
        sub_manager = SubscriptionManager()
        username = st.session_state.username
        
        # Use NEW method name
        remaining = sub_manager.get_ai_questions_remaining(username)
        subscription = sub_manager.get_user_subscription(username)
        tier = subscription.get('tier', 'free')
        
        # Show remaining questions
        if tier == 'free':
            if remaining > 0:
                st.info(f"🎁 You have {remaining} free AI question{'s' if remaining != 1 else ''} remaining (lifetime)")
            else:
                st.error("❌ You've used all your free AI questions (2 lifetime limit)")
                st.warning("Upgrade to Pro for unlimited AI chat!")
                if st.button("Upgrade to Pro", key="upgrade_chat", type="primary"):
                    st.session_state.show_pricing = True
                    st.rerun()
                return
        else:
            st.success("✓ Pro Plan - Unlimited AI questions")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if remaining > 0:
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
                            
                            # Increment and get new remaining
                            sub_manager.increment_ai_questions(username)
                            new_remaining = sub_manager.get_ai_questions_remaining(username)
                            
                            if new_remaining == 0 and tier == 'free':
                                st.warning("⚠️ That was your last free question! Upgrade to Pro for unlimited AI chat.")
                                st.balloons()
                            
                            st.rerun()
                            
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            logger.error(error_msg)
    
    except Exception as e:
        st.error(f"Error initializing chat: {str(e)}")
        st.info("Please try refreshing the page.")

def subscription_interface():
    """Subscription and billing interface"""
    st.header("Subscription & Billing")
    
    from src.utils.subscription import SubscriptionManager
    from src.utils.stripe_handler import StripeHandler
    
    sub_manager = SubscriptionManager()
    stripe_handler = StripeHandler()
    
    username = st.session_state.username
    subscription = sub_manager.get_user_subscription(username)
    
    current_tier = subscription.get('tier', 'free')
    
    st.subheader("Current Plan")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tier_info = sub_manager.TIERS[current_tier]
        st.write(f"**{tier_info['name']} Plan**")
        
        if current_tier == 'pro':
            st.success("✓ Pro Subscription Active")
            if subscription.get('expires_at'):
                st.write(f"Renews: {subscription['expires_at'][:10]}")
        else:
            st.info("Free Plan")
            ai_used = subscription.get('ai_questions_used', 0)
            ai_limit = sub_manager.TIERS['free']['limits']['ai_questions']
            st.write(f"AI Questions Used: {ai_used}/{ai_limit}")
    
    with col2:
        if current_tier == 'free':
            if st.button("Upgrade to Pro", key="upgrade_button", type="primary"):
                st.session_state.show_pricing = True
        else:
            st.write(f"**${tier_info['price']:.2f}/month**")
    
    st.markdown("---")
    
    if current_tier == 'free' or st.session_state.get('show_pricing', False):
        st.subheader("Compare Plans")
        
        col_free, col_pro = st.columns(2)
        
        with col_free:
            st.write("### Free")
            st.write("**$0/month**")
            st.write("")
            for feature in sub_manager.TIERS['free']['features']:
                st.write(f"✓ {feature}")
            st.write("")
            st.caption("Perfect for getting started")
        
        with col_pro:
            st.write("### Pro")
            st.write("**$24.99/month**")
            st.write("")
            for feature in sub_manager.TIERS['pro']['features']:
                st.write(f"✓ {feature}")
            st.write("")
            
            if current_tier == 'free':
                if stripe_handler.is_configured():
                    user_email = st.session_state.get('user_info', {}).get('email', '')
                    
                    if not user_email:
                        user_email = st.text_input("Email for billing:", key="billing_email")
                    
                    if user_email and st.button("Subscribe Now", key="subscribe_pro_button", type="primary"):
                       price_id = os.getenv('STRIPE_PRO_PRICE_ID', '')
                        
                        if not price_id or price_id == 'price_xxxxx':
                            st.error("Stripe Price ID not configured. Please contact support.")
                            st.info("Admin: Add STRIPE_PRO_PRICE_ID to Streamlit secrets")
                        else:
                            checkout_url = stripe_handler.create_checkout_session(
                                username,
                                user_email,
                                price_id
                            )
                            
                            if checkout_url:
                                st.markdown(stripe_handler.get_checkout_button_html(checkout_url), unsafe_allow_html=True)
                                st.info("Click the button above to complete your subscription.")
                else:
                    st.warning("Payment processing is currently in setup mode.")
                    st.info("Contact support@aidatachat.com to upgrade to Pro")
                    st.caption("Admin: Configure Stripe keys in Streamlit secrets to enable payments")
            else:
                st.success("✓ Current Plan")
    
    st.markdown("---")
    st.subheader("Feature Access")
    
    features_status = {
        'Time-Series Forecasting': sub_manager.can_access_feature(username, 'forecasting'),
        'Scenario Simulation': sub_manager.can_access_feature(username, 'scenarios'),
        'Data Quality Trust Score': sub_manager.can_access_feature(username, 'trust_score'),
        'Narrative Reports': sub_manager.can_access_feature(username, 'reports'),
    }
    
    for feature, has_access in features_status.items():
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.write(feature)
        with col_b:
            if has_access:
                st.success("✓ Enabled")
            else:
                st.error("✗ Pro Only")
    
    if current_tier == 'free':
        st.markdown("---")
        st.subheader("Usage Statistics")
        
        ai_used = subscription.get('ai_questions_used', 0)
        ai_limit = sub_manager.TIERS['free']['limits']['ai_questions']
        ai_remaining = sub_manager.get_ai_questions_remaining(username)
        
        progress_value = ai_used / ai_limit if ai_limit > 0 else 0
        
        st.write("**AI Chat Questions:**")
        st.progress(progress_value, text=f"{ai_used} of {ai_limit} used ({ai_remaining} remaining)")
        
        if ai_remaining == 0:
            st.error("⚠️ You've used all your free AI questions. Upgrade to Pro for unlimited access!")
        elif ai_remaining == 1:
            st.warning("⚠️ Only 1 free question remaining. Consider upgrading!")


def main():
    """Main application"""
    initialize_session_state()
    
    if not st.session_state.authenticated:
        login_page()
        return
    
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .tagline {
            font-size: 1.2rem;
            color: #888;
            text-align: center;
            margin-bottom: 2rem;
            font-style: italic;
        }
        .user-info {
            text-align: right;
            color: #666;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    user_name = st.session_state.get('user_info', {}).get("name", st.session_state.get('username', 'User'))
    st.markdown(f'<p class="user-info">Logged in as: <strong>{user_name}</strong></p>', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">AI DataChat</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Universal Intelligence Through Data</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("Upload Data")
        
        if st.button("Logout", key="logout_button", use_container_width=True):
            logout()
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload CSV, Excel, or JSON files (max 2GB)"
        )
        
        if uploaded_file:
            if st.button("Load Data", key="load_data_button"):
                with st.spinner("Loading data..."):
                    if load_data_file(uploaded_file):
                        st.success("Data loaded successfully!")
        
        if st.session_state.df is not None:
            st.markdown("---")
            st.subheader("File Info")
            st.write(f"**Name:** {st.session_state.file_info['name']}")
            st.write(f"**Format:** {st.session_state.file_info['format']}")
            st.write(f"**Size:** {st.session_state.file_info['size_mb']} MB")
            st.write(f"**Rows:** {len(st.session_state.df):,}")
            
            st.markdown("---")
            data_filtering_interface()
            
            if st.button("Reset Filters", key="reset_filters_button"):
                st.session_state.df_filtered = st.session_state.df.copy()
                st.rerun()
            
            if st.button("Clear Data", key="clear_data_button"):
                for key in ['df', 'df_filtered', 'file_info', 'data_summary', 'ai_agent', 'chat_history', 'narrative_report', 'html_report']:
                    st.session_state[key] = None if key != 'chat_history' else []
                st.rerun()
    
    if st.session_state.df is None:
        st.info("Upload a file to get started!")
        
        st.subheader("Welcome to AI DataChat")
        st.write("Transform your data into actionable insights with AI-powered analytics.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**What you can do:**")
            st.write("- Upload and analyze CSV, Excel, JSON files")
            st.write("- Interactive visualizations and statistics")
            st.write("- Data cleaning and filtering")
            st.write("- AI-powered chat with your data")
        
        with col2:
            st.write("**Pro Features:**")
            st.write("- Time-series forecasting (6-month predictions)")
            st.write("- Scenario simulation (what-if analysis)")
            st.write("- Data quality trust scores")
            st.write("- Automated narrative reports")
            st.write("- Unlimited AI questions")
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Statistics", "Forecasting", "Scenarios", "Chat", "Subscription"])
        
        with tab1:
            display_data_overview()
        
        with tab2:
            display_statistics()
        
        with tab3:
            forecasting_interface()
        
        with tab4:
            scenario_simulation_interface()
        
        with tab5:
            chat_interface()
        
        with tab6:
            subscription_interface()


if __name__ == "__main__":
    main()