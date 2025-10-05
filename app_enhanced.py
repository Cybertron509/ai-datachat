"""
AI DataChat Pro - Enhanced Application with Advanced Features
Complete Phase 2: Visualizations, Statistical Analysis, Export
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import custom modules
try:
    from src.utils.file_handler import FileHandler
    from src.utils.data_analyzer import DataAnalyzer
    from src.utils.visualizer import Visualizer
    from src.utils.exporter import Exporter
    from src.utils.statistical_analyzer import StatisticalAnalyzer
    from src.agents.ai_agent import AIAgent
    from config.settings import settings
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info("Make sure all required files are in the correct directories")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI DataChat Pro",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'ai_agent' not in st.session_state:
    st.session_state.ai_agent = AIAgent()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">💬 AI DataChat Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Your Advanced Conversational Data Analysis Assistant")
    st.markdown("---")
    
    # Sidebar
    render_sidebar()
    
    # Main content area
    if not st.session_state.data_loaded:
        render_welcome_screen()
    else:
        render_main_interface()


def render_sidebar():
    """Render sidebar with file upload and controls"""
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=FileHandler.get_supported_extensions(),
            help="Upload CSV, Excel, or JSON files"
        )
        
        if uploaded_file is not None:
            if not FileHandler.validate_file_size(uploaded_file, settings.max_file_size_mb):
                st.error(f"File size exceeds {settings.max_file_size_mb}MB limit!")
                return
            
            try:
                with st.spinner("Loading file..."):
                    result = FileHandler.load_file(uploaded_file)
                    
                    if isinstance(result, dict):
                        st.info(f"Excel file has {len(result)} sheets")
                        sheet_name = st.selectbox("Select sheet:", list(result.keys()))
                        st.session_state.dataframe = result[sheet_name]
                    else:
                        st.session_state.dataframe = result
                    
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.data_loaded = True
                    
                    summary = DataAnalyzer.generate_natural_language_summary(st.session_state.dataframe)
                    st.session_state.ai_agent.add_system_context(summary)
                    
                    st.success(f"✅ Loaded: {uploaded_file.name}")
            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        
        if st.session_state.data_loaded:
            st.markdown("---")
            st.subheader("📊 Data Info")
            df = st.session_state.dataframe
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.metric("Missing", f"{df.isnull().sum().sum():,}")
            
            st.markdown("---")
            st.subheader("⚙️ Actions")
            
            if st.button("🗑️ Clear Data", use_container_width=True):
                st.session_state.dataframe = None
                st.session_state.file_name = None
                st.session_state.data_loaded = False
                st.session_state.chat_history = []
                st.session_state.ai_agent.clear_history()
                st.rerun()
            
            if st.button("🔄 Reset Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.ai_agent.clear_history()
                summary = DataAnalyzer.generate_natural_language_summary(st.session_state.dataframe)
                st.session_state.ai_agent.add_system_context(summary)
                st.success("Chat history cleared!")


def render_welcome_screen():
    """Render welcome screen when no data is loaded"""
    st.info("👈 Upload a file to get started!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎉 Welcome to AI DataChat Pro!
        
        **Core Features:**
        - 📤 Upload CSV, Excel, or JSON files
        - 💬 Chat with your data using AI
        - 📊 Get instant insights and analysis
        - 📈 Interactive visualizations
        
        **Advanced Features:**
        - 📉 Statistical analysis tools
        - 📋 Export reports and data
        - 🎨 10+ visualization types
        - 🔬 Outlier detection & more
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Getting Started
        
        **Step 1:** Upload your data file
        - Supported: CSV, Excel, JSON
        - Max size: 100MB
        
        **Step 2:** Explore your data
        - View data overview
        - Check data quality
        
        **Step 3:** Analyze with AI
        - Ask questions naturally
        - Get visualization tips
        
        **Step 4:** Export results
        - Download cleaned data
        - Generate reports
        """)


def render_main_interface():
    """Render main interface with tabs"""
    tabs = st.tabs([
        "💬 Chat", 
        "📊 Data Overview", 
        "📈 Visualizations", 
        "🔬 Statistical Analysis",
        "📋 Export"
    ])
    
    with tabs[0]:
        show_chat_interface()
    
    with tabs[1]:
        show_data_overview()
    
    with tabs[2]:
        show_visualizations()
    
    with tabs[3]:
        show_statistical_analysis()
    
    with tabs[4]:
        show_export_interface()


def show_chat_interface():
    """Display chat interface"""
    st.subheader("💬 Chat with Your Data")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask anything about your data..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.ai_agent.chat(prompt)
                st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    st.markdown("---")
    st.markdown("**Quick Actions:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Summarize", use_container_width=True):
            summary = DataAnalyzer.generate_natural_language_summary(st.session_state.dataframe)
            st.session_state.chat_history.append({"role": "user", "content": "Summarize the data"})
            st.session_state.chat_history.append({"role": "assistant", "content": summary})
            st.rerun()
    
    with col2:
        if st.button("📈 Visualizations", use_container_width=True):
            column_types = DataAnalyzer.detect_column_types(st.session_state.dataframe)
            response = st.session_state.ai_agent.suggest_visualizations(column_types)
            st.session_state.chat_history.append({"role": "user", "content": "Suggest visualizations"})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col3:
        if st.button("🔍 Find Issues", use_container_width=True):
            quality_report = DataAnalyzer.generate_data_quality_report(st.session_state.dataframe)
            issues = [f"- {col}: {report['missing_percentage']:.1f}% missing" 
                     for col, report in quality_report.items() if report['missing_percentage'] > 10]
            response = "Data quality issues:\n" + "\n".join(issues) if issues else "No significant issues found! ✅"
            st.session_state.chat_history.append({"role": "user", "content": "Find issues"})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col4:
        if st.button("🔗 Correlations", use_container_width=True):
            correlations = DataAnalyzer.find_correlations(st.session_state.dataframe, threshold=0.7)
            if correlations:
                response = "Strong correlations:\n" + "\n".join([
                    f"- {c['column1']} ↔ {c['column2']}: {c['correlation']:.3f}"
                    for c in correlations[:5]
                ])
            else:
                response = "No strong correlations found"
            st.session_state.chat_history.append({"role": "user", "content": "Find correlations"})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()


def show_data_overview():
    """Display data overview"""
    df = st.session_state.dataframe
    
    st.subheader(f"📊 Data Overview: {st.session_state.file_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        st.metric("Duplicates", f"{df.duplicated().sum():,}")
    
    st.markdown("---")
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📋 Column Information")
    
    column_info = [{
        "Column": col,
        "Type": str(df[col].dtype),
        "Missing": df[col].isnull().sum(),
        "Missing %": f"{df[col].isnull().sum() / len(df) * 100:.1f}%",
        "Unique": df[col].nunique()
    } for col in df.columns]
    
    st.dataframe(pd.DataFrame(column_info), use_container_width=True, hide_index=True)


def show_visualizations():
    """Display visualization interface"""
    df = st.session_state.dataframe
    
    st.subheader("📈 Interactive Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Correlation Heatmap", "Distribution", "Scatter Plot", "Line Chart", 
         "Bar Chart", "Box Plot"]
    )
    
    try:
        if viz_type == "Correlation Heatmap":
            fig = Visualizer.create_correlation_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns")
        
        elif viz_type == "Distribution":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select column", numeric_cols)
                bins = st.slider("Bins", 10, 100, 30)
                fig = Visualizer.create_histogram(df, col, bins)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plot":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
                fig = Visualizer.create_scatter_plot(df, x_col, y_col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Chart":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            all_cols = df.columns.tolist()
            if numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", all_cols)
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols)
                fig = Visualizer.create_line_chart(df, x_col, y_col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", df.columns.tolist())
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols)
                fig = Visualizer.create_bar_chart(df, x_col, y_col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                y_col = st.selectbox("Select column", numeric_cols)
                fig = Visualizer.create_box_plot(df, y_col)
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")


def show_statistical_analysis():
    """Display statistical analysis"""
    df = st.session_state.dataframe
    
    st.subheader("🔬 Statistical Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis",
        ["Outlier Detection", "Normality Test", "Confidence Interval", "Correlation Test"]
    )
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    try:
        if analysis_type == "Outlier Detection" and numeric_cols:
            col = st.selectbox("Select column", numeric_cols)
            if st.button("Detect Outliers"):
                result = StatisticalAnalyzer.detect_outliers_iqr(df, col)
                if "error" not in result:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total", result['total_values'])
                    with col2:
                        st.metric("Outliers", result['outlier_count'])
                    with col3:
                        st.metric("Percentage", f"{result['outlier_percentage']:.2f}%")
                    fig = Visualizer.create_box_plot(df, col)
                    st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Normality Test" and numeric_cols:
            col = st.selectbox("Select column", numeric_cols)
            if st.button("Test"):
                result = StatisticalAnalyzer.perform_normality_test(df, col)
                if "error" not in result:
                    st.success(result['interpretation'])
                    st.metric("P-Value", f"{result['p_value']:.4f}")
        
        elif analysis_type == "Confidence Interval" and numeric_cols:
            col = st.selectbox("Select column", numeric_cols)
            confidence = st.slider("Confidence", 0.90, 0.99, 0.95, 0.01)
            if st.button("Calculate"):
                result = StatisticalAnalyzer.calculate_confidence_interval(df, col, confidence)
                if "error" not in result:
                    st.metric("Mean", f"{result['mean']:.2f}")
                    st.write(f"CI: [{result['lower_bound']:.2f}, {result['upper_bound']:.2f}]")
        
        elif analysis_type == "Correlation Test" and len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("First", numeric_cols)
            with col2:
                y_col = st.selectbox("Second", [c for c in numeric_cols if c != x_col])
            if st.button("Test"):
                result = StatisticalAnalyzer.perform_correlation_test(df, x_col, y_col)
                if "error" not in result:
                    st.metric("Correlation", f"{result['pearson_correlation']:.3f}")
                    st.write(f"Strength: {result['pearson_strength']}")
    
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")


def show_export_interface():
    """Display export interface"""
    df = st.session_state.dataframe
    
    st.subheader("📋 Export Your Data")
    
    export_format = st.radio("Format", ["CSV", "Excel", "JSON"], horizontal=True)
    
    if st.button(f"📥 Download as {export_format}", use_container_width=True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format == "CSV":
            data = Exporter.export_to_csv(df)
            filename = f"data_{timestamp}.csv"
            mime = "text/csv"
        elif export_format == "Excel":
            data = Exporter.export_to_excel(df)
            filename = f"data_{timestamp}.xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            data = Exporter.export_to_json(df)
            filename = f"data_{timestamp}.json"
            mime = "application/json"
        
        st.download_button(
            label=f"💾 Download {filename}",
            data=data,
            file_name=filename,
            mime=mime
        )
    
    st.markdown("---")
    st.subheader("📄 Generate Report")
    
    if st.button("Generate Analysis Report"):
        summary = DataAnalyzer.get_summary_statistics(df)
        quality_report = DataAnalyzer.generate_data_quality_report(df)
        report = Exporter.create_analysis_report(df, summary, quality_report, st.session_state.chat_history)
        
        st.markdown(report)
        
        st.download_button(
            "💾 Download Report",
            data=report.encode('utf-8'),
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )


if __name__ == "__main__":
    main()