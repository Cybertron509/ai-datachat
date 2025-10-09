"""
Data Quality & Trust Score Analyzer
Evaluates uploaded data and provides reliability metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DataQualityAnalyzer:
    """Analyze data quality and generate trust scores"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []
        self.warnings = []
        self.info_items = []
        
    def calculate_trust_score(self) -> Dict:
        """Calculate overall trust score (0-100) and detailed metrics"""
        
        scores = {
            'completeness': self._check_completeness(),
            'consistency': self._check_consistency(),
            'uniqueness': self._check_uniqueness(),
            'validity': self._check_validity(),
            'accuracy': self._check_accuracy()
        }
        
        # Weighted average (completeness and validity are most important)
        weights = {
            'completeness': 0.30,
            'consistency': 0.20,
            'uniqueness': 0.15,
            'validity': 0.25,
            'accuracy': 0.10
        }
        
        trust_score = sum(scores[key] * weights[key] for key in scores)
        
        return {
            'overall_score': round(trust_score, 1),
            'component_scores': scores,
            'issues': self.issues,
            'warnings': self.warnings,
            'info': self.info_items,
            'grade': self._get_grade(trust_score),
            'recommendations': self._get_recommendations(scores)
        }
    
    def _check_completeness(self) -> float:
        """Check for missing values"""
        total_cells = self.df.size
        missing_cells = self.df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        
        if missing_pct > 20:
            self.issues.append(f"High missing data: {missing_pct:.1f}% of cells are empty")
        elif missing_pct > 5:
            self.warnings.append(f"Moderate missing data: {missing_pct:.1f}% of cells are empty")
        else:
            self.info_items.append(f"Low missing data: {missing_pct:.1f}% of cells are empty")
        
        # Score: 100 for 0% missing, 0 for 50%+ missing
        score = max(0, 100 - (missing_pct * 2))
        return score
    
    def _check_consistency(self) -> float:
        """Check data type consistency and format issues"""
        inconsistencies = 0
        total_checks = 0
        
        for col in self.df.columns:
            total_checks += 1
            
            # Check if numeric columns have string contamination
            if self.df[col].dtype in ['int64', 'float64']:
                continue
            
            # Check if supposedly numeric data is stored as strings
            if self.df[col].dtype == 'object':
                # Try converting to numeric
                try:
                    numeric_version = pd.to_numeric(self.df[col], errors='coerce')
                    numeric_count = numeric_version.notna().sum()
                    total_count = self.df[col].notna().sum()
                    
                    if numeric_count / max(total_count, 1) > 0.8:
                        inconsistencies += 1
                        self.warnings.append(f"Column '{col}' appears numeric but stored as text")
                except:
                    pass
        
        # Check for mixed date formats
        for col in self.df.select_dtypes(include=['object']).columns:
            sample = self.df[col].dropna().head(100)
            if len(sample) > 0:
                # Simple date pattern check
                date_like = sample.astype(str).str.contains(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', regex=True).sum()
                if date_like > len(sample) * 0.5:
                    self.info_items.append(f"Column '{col}' may contain dates - consider converting")
        
        score = max(0, 100 - (inconsistencies / max(total_checks, 1) * 100))
        return score
    
    def _check_uniqueness(self) -> float:
        """Check for duplicate rows"""
        duplicate_count = self.df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(self.df)) * 100
        
        if duplicate_count > 0:
            if duplicate_pct > 5:
                self.issues.append(f"Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)")
            else:
                self.warnings.append(f"Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)")
        else:
            self.info_items.append("No duplicate rows detected")
        
        score = max(0, 100 - (duplicate_pct * 10))
        return score
    
    def _check_validity(self) -> float:
        """Check for invalid values (outliers, negatives where shouldn't be, etc.)"""
        issues_found = 0
        total_checks = 0
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            total_checks += 1
            
            # Check for negative values in likely positive-only fields
            if any(keyword in col.lower() for keyword in ['age', 'count', 'quantity', 'price', 'amount']):
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    issues_found += 1
                    self.warnings.append(f"Column '{col}' has {negative_count} negative values")
            
            # Check for outliers using IQR method
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < Q1 - 3 * IQR) | (self.df[col] > Q3 + 3 * IQR)).sum()
            outlier_pct = (outliers / len(self.df)) * 100
            
            if outlier_pct > 5:
                issues_found += 1
                self.warnings.append(f"Column '{col}' has {outliers} extreme outliers ({outlier_pct:.1f}%)")
        
        if total_checks == 0:
            return 100
        
        score = max(0, 100 - (issues_found / total_checks * 50))
        return score
    
    def _check_accuracy(self) -> float:
        """Check for data accuracy indicators"""
        issues = 0
        total_checks = 0
        
        # Check for suspicious patterns
        for col in self.df.columns:
            total_checks += 1
            
            # Check for columns with single value (likely data entry errors)
            unique_count = self.df[col].nunique()
            if unique_count == 1 and len(self.df) > 10:
                issues += 1
                self.warnings.append(f"Column '{col}' has only one unique value - may be constant/error")
            
            # Check for suspiciously high cardinality in small datasets
            if len(self.df) < 100 and unique_count == len(self.df):
                if self.df[col].dtype == 'object':
                    self.info_items.append(f"Column '{col}' has unique values for every row - may be an ID")
        
        score = max(0, 100 - (issues / max(total_checks, 1) * 100))
        return score
    
    def _get_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return "A - Excellent"
        elif score >= 80:
            return "B - Good"
        elif score >= 70:
            return "C - Fair"
        elif score >= 60:
            return "D - Poor"
        else:
            return "F - Critical Issues"
    
    def _get_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on scores"""
        recommendations = []
        
        if scores['completeness'] < 80:
            recommendations.append("Remove or impute missing values in the 'Statistics' tab")
        
        if scores['uniqueness'] < 90:
            recommendations.append("Remove duplicate rows to improve data quality")
        
        if scores['validity'] < 80:
            recommendations.append("Review outliers and negative values in numeric columns")
        
        if scores['consistency'] < 85:
            recommendations.append("Check data types - some columns may need conversion")
        
        if not recommendations:
            recommendations.append("Data quality is good! No major issues detected.")
        
        return recommendations