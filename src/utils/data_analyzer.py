"""Data analysis utilities"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Analyze and generate insights from DataFrames"""
    
    @staticmethod
    def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive summary statistics for a DataFrame"""
        summary = {
            "shape": {
                "rows": int(len(df)),
                "columns": int(len(df.columns))
            },
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / (1024**2)), 2),
            "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
            "missing_percentage": {col: round(float(df[col].isnull().sum() / len(df) * 100), 2) for col in df.columns}
        }
        return summary
    
    @staticmethod
    def get_numeric_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Get statistics for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {}
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum())
            }
        return stats
    
    @staticmethod
    def get_categorical_statistics(df: pd.DataFrame, max_categories: int = 50) -> Dict[str, Dict[str, Any]]:
        """Get statistics for categorical columns"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {}
        
        stats = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            
            if len(value_counts) > max_categories:
                value_counts = value_counts.head(max_categories)
            
            stats[col] = {
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "most_common": str(df[col].mode()[0]) if not df[col].mode().empty else None,
                "value_counts": {str(k): int(v) for k, v in value_counts.items()}
            }
        return stats
    
    @staticmethod
    def find_correlations(df: pd.DataFrame, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find significant correlations between numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return []
        
        corr_matrix = numeric_df.corr()
        correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": round(float(corr_value), 3),
                        "strength": DataAnalyzer._correlation_strength(corr_value)
                    })
        
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return correlations
    
    @staticmethod
    def _correlation_strength(corr: float) -> str:
        """Classify correlation strength"""
        abs_corr = abs(corr)
        if abs_corr >= 0.9:
            return "very strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very weak"