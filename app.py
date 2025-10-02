"""
AI DataChat - Enhanced Main Application with Authentication and Rate Limiting
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

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.file_handler import FileHandler
from src.utils.data_analyzer import DataAnalyzer
from src.agents.ai_agent import AIAgent
from src.utils.auth import AuthManager
from src.utils.rate_limiter import RateLimiter
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
    """Create interactive visualizations with analytical guidance and performance optimization"""
    st.subheader("Interactive Visualizations")
    
    # Performance warning for large datasets
    if len(df) > 10000:
        st.warning(f"Large dataset detected ({len(df):,} rows). Some visualizations will use sampling for performance.")
    
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Histogram", "Scatter Plot", "Bar Chart", "Box Plot", "Correlation Heatmap", "Line Chart"],
        key="viz_type_selector"
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Limit columns for large datasets
    max_cols_for_viz = 20
    if len(numeric_cols) > max_cols_for_viz:
        st.info(f"Dataset has {len(numeric_cols)} numeric columns. Selecting top {max_cols_for_viz} most variable for performance.")
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(max_cols_for_viz).index.tolist()
    
    if viz_type == "Histogram":
        if numeric_cols:
            col = st.selectbox("Select column:", numeric_cols, key="hist_column")
            st.info(f"Showing distribution of {col}. Look for patterns, outliers, or skewness.")
            
            # Sample for very large datasets
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
            
            # Limit color options for performance
            color_options = ["None"] + cat_cols[:5] + numeric_cols[:5]
            color_col = st.selectbox("Color by (optional):", color_options, key="scatter_color")
            color_col = None if color_col == "None" else color_col
            
            # Always sample scatter plots for performance
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
            
            # Limit categories for readability
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
            
            # Sample for large datasets
            plot_df = df if len(df) <= 10000 else df.sample(10000)
            if len(df) > 10000:
                st.caption(f"Showing sample of 10,000 from {len(df):,} total rows")
            
            if group_by == "None":
                fig = px.box(plot_df, y=col, title=f"Box Plot of {col}")
            else:
                # Limit groups
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
            
            # Limit columns for performance
            max_corr_cols = 15
            if len(numeric_cols) > max_corr_cols:
                st.warning(f"Showing top {max_corr_cols} most variable columns for clarity.")
                variances = df[numeric_cols].var().sort_values(ascending=False)
                selected_cols = variances.head(max_corr_cols).index.tolist()
            else:
                selected_cols = numeric_cols
            
            # Sample if dataset is large
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
            
            # Limit and sort data
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
    
    col1, col2, col3 = st.columns(3)
    
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


def display_data_overview():
    """Enhanced data overview with visualizations"""
    if st.session_state.df_filtered is None:
        return
    
    st.header("Data Overview")
    
    df = st.session_state.df_filtered
    
    # Metrics
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
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Visualizations
    create_visualizations(df)
    
    # Export options
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


def chat_interface():
    """AI chat interface with rate limiting"""
    st.header("Chat with Your Data")
    
    if st.session_state.df_filtered is None:
        st.info("Please load data first")
        return
    
    if st.session_state.ai_agent is None:
        st.warning("AI Chat is not available. Please check your OpenAI API configuration and ensure you have sufficient credits.")
        return
    
    # Display remaining queries
    remaining = RateLimiter.get_remaining_queries()
    if remaining > 0:
        st.info(f"You have {remaining} question{'s' if remaining != 1 else ''} remaining")
    else:
        st.error("You've reached your question limit (2 questions per session). Reload the page or clear data to reset.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input - only if queries remain
    if RateLimiter.can_query():
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
                        
                        # Increment query count
                        RateLimiter.increment_query()
                        
                        # Show updated count
                        remaining = RateLimiter.get_remaining_queries()
                        if remaining == 0:
                            st.warning("That was your last question. Reload the page or clear data to reset.")
                        else:
                            st.rerun()
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        logger.error(error_msg)


def main():
    """Main application"""
    initialize_session_state()
    
    # Check authentication
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Custom CSS
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
    
    # User info
    user_name = st.session_state.user_info.get("name", st.session_state.username)
    st.markdown(f'<p class="user-info">Logged in as: <strong>{user_name}</strong></p>', unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">AI DataChat</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Universal Intelligence Through Data</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
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
                for key in ['df', 'df_filtered', 'file_info', 'data_summary', 'ai_agent', 'chat_history']:
                    st.session_state[key] = None if key != 'chat_history' else []
                st.rerun()
    
    # Main content
    if st.session_state.df is None:
        st.info("Upload a file to get started!")
        
        st.subheader("Sample Datasets Available")
        st.write("You can find sample datasets in: `data/samples/`")
    else:
        tab1, tab2, tab3 = st.tabs(["Overview", "Statistics", "Chat"])
        
        with tab1:
            display_data_overview()
        
        with tab2:
            display_statistics()
        
        with tab3:
            chat_interface()


if __name__ == "__main__":
    main()