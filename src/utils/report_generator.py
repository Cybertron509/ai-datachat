"""
Narrative Report Generator
Creates executive summaries with AI-written insights
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List
import base64
from io import BytesIO


class ReportGenerator:
    """Generate narrative reports from data analysis"""
    
    def __init__(self, df: pd.DataFrame, quality_report: Dict, data_summary: Dict):
        self.df = df
        self.quality_report = quality_report
        self.data_summary = data_summary
        
    def generate_executive_summary(self) -> str:
        """Generate narrative executive summary"""
        
        summary_parts = []
        
        # Header
        summary_parts.append(f"# Data Analysis Executive Summary")
        summary_parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_parts.append("")
        
        # Dataset Overview
        summary_parts.append("## Dataset Overview")
        summary_parts.append(f"This analysis examines a dataset containing **{len(self.df):,} records** across **{len(self.df.columns)} variables**.")
        
        # Memory and completeness
        memory_mb = self.df.memory_usage(deep=True).sum() / (1024**2)
        missing_pct = (self.df.isnull().sum().sum() / self.df.size) * 100
        summary_parts.append(f"The dataset occupies {memory_mb:.2f} MB of memory with {missing_pct:.1f}% missing values.")
        summary_parts.append("")
        
        # Data Quality Assessment
        summary_parts.append("## Data Quality Assessment")
        score = self.quality_report['overall_score']
        grade = self.quality_report['grade']
        summary_parts.append(f"**Trust Score: {score}/100 ({grade})**")
        summary_parts.append("")
        
        # Interpret trust score
        if score >= 90:
            summary_parts.append("The data demonstrates excellent quality with minimal issues. Analysis results can be considered highly reliable.")
        elif score >= 80:
            summary_parts.append("The data quality is good with some minor areas for improvement. Results are generally reliable for decision-making.")
        elif score >= 70:
            summary_parts.append("The data quality is fair. Some data cleaning is recommended before making critical business decisions.")
        elif score >= 60:
            summary_parts.append("The data has quality concerns that should be addressed. Proceed with caution when using these insights for decision-making.")
        else:
            summary_parts.append("The data has significant quality issues. Substantial cleaning and validation are required before analysis results can be trusted.")
        summary_parts.append("")
        
        # Key Issues
        if self.quality_report['issues'] or self.quality_report['warnings']:
            summary_parts.append("### Key Findings:")
            if self.quality_report['issues']:
                for issue in self.quality_report['issues']:
                    summary_parts.append(f"- ⚠️ {issue}")
            if self.quality_report['warnings']:
                for warning in self.quality_report['warnings']:
                    summary_parts.append(f"- ⚡ {warning}")
            summary_parts.append("")
        
        # Statistical Insights
        summary_parts.append("## Statistical Insights")
        
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            summary_parts.append("### Numeric Variables")
            for col in numeric_cols[:5]:  # Top 5 numeric columns
                mean_val = self.df[col].mean()
                median_val = self.df[col].median()
                std_val = self.df[col].std()
                
                summary_parts.append(f"**{col}:**")
                summary_parts.append(f"- Average: {mean_val:.2f}, Median: {median_val:.2f}, Std Dev: {std_val:.2f}")
                
                # Interpret distribution
                if abs(mean_val - median_val) / (std_val + 0.001) > 0.5:
                    summary_parts.append(f"- Distribution appears skewed (mean differs significantly from median)")
                else:
                    summary_parts.append(f"- Distribution appears relatively symmetric")
                summary_parts.append("")
        
        # Correlations
        if len(numeric_cols) >= 2:
            summary_parts.append("### Key Relationships")
            corr_matrix = self.df[numeric_cols].corr()
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))
            
            if corr_pairs:
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                for var1, var2, corr_val in corr_pairs[:3]:
                    direction = "positive" if corr_val > 0 else "negative"
                    strength = "strong" if abs(corr_val) > 0.7 else "moderate"
                    summary_parts.append(f"- **{var1}** and **{var2}** show a {strength} {direction} correlation ({corr_val:.3f})")
                summary_parts.append("")
            else:
                summary_parts.append("No strong correlations detected between numeric variables.")
                summary_parts.append("")
        
        # Categorical insights
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            summary_parts.append("### Categorical Variables")
            for col in cat_cols[:3]:  # Top 3 categorical
                unique_count = self.df[col].nunique()
                most_common = self.df[col].mode()[0] if not self.df[col].mode().empty else "N/A"
                most_common_pct = (self.df[col] == most_common).sum() / len(self.df) * 100
                
                summary_parts.append(f"**{col}:** {unique_count} unique values")
                summary_parts.append(f"- Most common: '{most_common}' ({most_common_pct:.1f}% of records)")
                summary_parts.append("")
        
        # Recommendations
        summary_parts.append("## Recommendations")
        for rec in self.quality_report['recommendations']:
            summary_parts.append(f"- {rec}")
        summary_parts.append("")
        
        # Conclusion
        summary_parts.append("## Conclusion")
        if score >= 80:
            summary_parts.append("This dataset is suitable for analysis and decision-making. The insights generated are reliable and can inform strategic planning.")
        elif score >= 60:
            summary_parts.append("While this dataset provides useful insights, addressing the identified quality issues will improve the reliability of conclusions drawn from this analysis.")
        else:
            summary_parts.append("Significant data quality improvements are needed before this dataset can reliably support decision-making. Focus on addressing critical issues first.")
        
        return "\n".join(summary_parts)
    
    def generate_html_report(self) -> str:
        """Generate HTML version of report with embedded charts"""
        
        html_parts = []
        
        # HTML Header
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Data Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1 { color: #1f77b4; border-bottom: 3px solid #1f77b4; }
                h2 { color: #2c5282; margin-top: 30px; }
                h3 { color: #4a5568; }
                .metric { background: #f7fafc; padding: 15px; border-radius: 8px; margin: 10px 0; }
                .score { font-size: 48px; font-weight: bold; text-align: center; margin: 20px 0; }
                .good { color: green; }
                .fair { color: orange; }
                .poor { color: red; }
                .chart { margin: 20px 0; }
                ul { line-height: 1.8; }
            </style>
        </head>
        <body>
        """)
        
        # Title
        html_parts.append(f"<h1>Data Analysis Executive Summary</h1>")
        html_parts.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Trust Score
        score = self.quality_report['overall_score']
        score_class = "good" if score >= 80 else "fair" if score >= 60 else "poor"
        html_parts.append(f"<div class='metric'>")
        html_parts.append(f"<h2>Data Quality Trust Score</h2>")
        html_parts.append(f"<div class='score {score_class}'>{score}/100</div>")
        html_parts.append(f"<p style='text-align: center;'>{self.quality_report['grade']}</p>")
        html_parts.append(f"</div>")
        
        # Dataset Overview
        html_parts.append(f"<h2>Dataset Overview</h2>")
        html_parts.append(f"<ul>")
        html_parts.append(f"<li>Records: {len(self.df):,}</li>")
        html_parts.append(f"<li>Variables: {len(self.df.columns)}</li>")
        html_parts.append(f"<li>Missing Values: {self.df.isnull().sum().sum():,}</li>")
        html_parts.append(f"</ul>")
        
        # Generate simple chart (top 5 numeric variables)
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()[:5]
        if numeric_cols:
            html_parts.append(f"<h2>Key Numeric Variables - Distribution</h2>")
            for col in numeric_cols:
                fig = px.histogram(self.df, x=col, title=f"Distribution of {col}")
                chart_html = fig.to_html(include_plotlyjs='cdn', div_id=f"chart_{col}")
                html_parts.append(chart_html)
        
        # Recommendations
        html_parts.append(f"<h2>Recommendations</h2>")
        html_parts.append(f"<ul>")
        for rec in self.quality_report['recommendations']:
            html_parts.append(f"<li>{rec}</li>")
        html_parts.append(f"</ul>")
        
        # Footer
        html_parts.append("""
        <hr>
        <p style='text-align: center; color: #888; font-size: 12px;'>
        Generated by AI DataChat - Universal Intelligence Through Data
        </p>
        </body>
        </html>
        """)
        
        return "".join(html_parts)
    
    def generate_markdown_report(self) -> str:
        """Generate markdown version for easy export"""
        return self.generate_executive_summary()