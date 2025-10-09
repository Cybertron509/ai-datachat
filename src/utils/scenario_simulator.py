"""
Scenario Simulation Engine
"What if" analysis with multi-variable changes and sensitivity analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


class ScenarioSimulator:
    """Simulate business scenarios with multi-variable changes"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.correlations = self._calculate_correlations()
        
    def _calculate_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix for numeric variables"""
        if len(self.numeric_cols) >= 2:
            return self.df[self.numeric_cols].corr()
        return pd.DataFrame()
    
    def get_correlated_variables(self, variable: str, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Find variables correlated with the given variable"""
        if variable not in self.correlations.columns:
            return []
        
        correlations = self.correlations[variable].drop(variable)
        # Filter by threshold and sort by absolute correlation
        strong_corr = correlations[abs(correlations) >= threshold]
        
        # Return sorted by absolute correlation strength
        return [(var, corr) for var, corr in strong_corr.items()]
    
    def parse_scenario_text(self, scenario_text: str) -> List[Dict]:
        """
        Parse simple scenario text into changes
        Examples:
        - "increase sales by 20%"
        - "increase revenue 15% and reduce costs 10%"
        - "decrease churn by 5%"
        """
        changes = []
        
        # Pattern: (increase|decrease|reduce) VARIABLE by NUMBER%
        pattern = r'(increase|decrease|reduce)\s+([a-zA-Z_]+)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%'
        
        matches = re.findall(pattern, scenario_text.lower())
        
        for action, variable, value in matches:
            # Find matching column (case-insensitive, partial match)
            matching_cols = [col for col in self.numeric_cols 
                           if variable.lower() in col.lower()]
            
            if matching_cols:
                # Use the first match
                col = matching_cols[0]
                percentage = float(value)
                
                if action in ['decrease', 'reduce']:
                    percentage = -percentage
                
                changes.append({
                    'variable': col,
                    'change_percent': percentage,
                    'action': action
                })
        
        return changes
    
    def simulate_scenario(
        self,
        changes: List[Dict],
        correlation_threshold: float = 0.3
    ) -> Dict:
        """
        Simulate a scenario with multiple variable changes
        
        Args:
            changes: List of dicts with 'variable' and 'change_percent'
            correlation_threshold: Minimum correlation to consider relationships
        
        Returns:
            Dict with simulation results
        """
        
        results = {
            'primary_changes': [],
            'secondary_effects': [],
            'summary': {}
        }
        
        # Calculate primary changes
        for change in changes:
            variable = change['variable']
            change_pct = change['change_percent']
            
            if variable not in self.numeric_cols:
                continue
            
            current_value = self.df[variable].mean()
            new_value = current_value * (1 + change_pct / 100)
            absolute_change = new_value - current_value
            
            results['primary_changes'].append({
                'variable': variable,
                'current_value': current_value,
                'new_value': new_value,
                'change_percent': change_pct,
                'absolute_change': absolute_change
            })
            
            # Find correlated variables (secondary effects)
            correlated = self.get_correlated_variables(variable, correlation_threshold)
            
            for corr_var, correlation in correlated:
                # Skip if this variable is being changed directly
                if any(c['variable'] == corr_var for c in changes):
                    continue
                
                # Estimate secondary effect based on correlation
                # Simplified model: correlated variable changes proportionally
                estimated_change_pct = change_pct * correlation
                
                corr_current = self.df[corr_var].mean()
                corr_new = corr_current * (1 + estimated_change_pct / 100)
                
                results['secondary_effects'].append({
                    'variable': corr_var,
                    'affected_by': variable,
                    'correlation': correlation,
                    'current_value': corr_current,
                    'estimated_value': corr_new,
                    'estimated_change_percent': estimated_change_pct,
                    'absolute_change': corr_new - corr_current
                })
        
        # Generate summary
        total_vars_affected = len(results['primary_changes']) + len(results['secondary_effects'])
        results['summary'] = {
            'total_variables_changed': len(results['primary_changes']),
            'total_secondary_effects': len(results['secondary_effects']),
            'total_variables_affected': total_vars_affected
        }
        
        return results
    
    def sensitivity_analysis(
        self,
        variable: str,
        base_change_percent: float,
        range_percent: float = 10
    ) -> Dict:
        """
        Perform sensitivity analysis around a base scenario
        
        Args:
            variable: Variable to analyze
            base_change_percent: Base percentage change
            range_percent: Range to test (e.g., ±10% around base)
        
        Returns:
            Dict with sensitivity results
        """
        
        if variable not in self.numeric_cols:
            return {'error': f'Variable {variable} not found'}
        
        # Test scenarios: base - range, base, base + range
        scenarios = [
            base_change_percent - range_percent,
            base_change_percent,
            base_change_percent + range_percent
        ]
        
        results = []
        
        for scenario_pct in scenarios:
            change = [{
                'variable': variable,
                'change_percent': scenario_pct
            }]
            
            sim_result = self.simulate_scenario(change)
            
            # Extract key metrics
            primary = sim_result['primary_changes'][0]
            
            results.append({
                'change_percent': scenario_pct,
                'new_value': primary['new_value'],
                'absolute_change': primary['absolute_change'],
                'secondary_effects_count': len(sim_result['secondary_effects'])
            })
        
        # Calculate sensitivity metric
        value_range = results[2]['new_value'] - results[0]['new_value']
        pct_range = scenarios[2] - scenarios[0]
        sensitivity = value_range / pct_range if pct_range != 0 else 0
        
        return {
            'variable': variable,
            'base_scenario': results[1],
            'pessimistic_scenario': results[0],
            'optimistic_scenario': results[2],
            'sensitivity_coefficient': sensitivity,
            'scenarios': results
        }
    
    def compare_scenarios(
        self,
        scenario_a: List[Dict],
        scenario_b: List[Dict]
    ) -> Dict:
        """Compare two different scenarios"""
        
        result_a = self.simulate_scenario(scenario_a)
        result_b = self.simulate_scenario(scenario_b)
        
        comparison = {
            'scenario_a': result_a,
            'scenario_b': result_b,
            'differences': []
        }
        
        # Compare primary changes
        for change_a in result_a['primary_changes']:
            var = change_a['variable']
            change_b = next((c for c in result_b['primary_changes'] if c['variable'] == var), None)
            
            if change_b:
                diff = change_b['new_value'] - change_a['new_value']
                comparison['differences'].append({
                    'variable': var,
                    'scenario_a_value': change_a['new_value'],
                    'scenario_b_value': change_b['new_value'],
                    'difference': diff
                })
        
        return comparison