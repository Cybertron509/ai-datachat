"""
AI Agent for data analysis using OpenAI
"""
import openai
import pandas as pd
import json
from typing import Dict, Any, List, Optional
import logging
from config import settings

logger = logging.getLogger(__name__)


class AIAgent:
    """AI agent for analyzing data and answering questions"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize AI Agent
        
        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model to use (defaults to settings)
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.api_key
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"Initialized AI Agent with model: {self.model}")
    
    def _create_system_prompt(self, df: pd.DataFrame, data_summary: Dict[str, Any]) -> str:
        """
        Create system prompt with data context
        
        Args:
            df: DataFrame being analyzed
            data_summary: Summary statistics of the data
            
        Returns:
            System prompt string
        """
        prompt = f"""You are an expert data analyst assistant. You help users understand and analyze their data.

**Dataset Information:**
- Rows: {data_summary['shape']['rows']}
- Columns: {data_summary['shape']['columns']}
- Column names: {', '.join(data_summary['columns'])}

**Column Data Types:**
{json.dumps(data_summary['dtypes'], indent=2)}

**Guidelines:**
1. Provide clear, concise answers about the data
2. When asked to analyze, provide specific insights with numbers
3. Suggest visualizations when appropriate
4. If calculation is needed, explain the approach
5. Be honest if the data doesn't contain information to answer the question
6. Format responses in a readable way with proper structure

**Available Information:**
You have access to the complete dataset and can reference any columns or statistics about the data.
"""
        return prompt
    
    def analyze_data(self, df: pd.DataFrame, question: str, data_summary: Dict[str, Any]) -> str:
        """
        Analyze data and answer a question
        
        Args:
            df: DataFrame to analyze
            question: User's question about the data
            data_summary: Summary statistics from DataAnalyzer
            
        Returns:
            AI-generated response
        """
        try:
            # Create system prompt with data context
            system_prompt = self._create_system_prompt(df, data_summary)
            
            # Add data sample to give context
            data_sample = df.head(5).to_string()
            context_message = f"Here's a sample of the data:\n{data_sample}"
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_message}
            ]
            
            # Add conversation history
            messages.extend(self.conversation_history)
            
            # Add current question
            messages.append({"role": "user", "content": question})
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            # Keep only last 10 messages to avoid token limits
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            logger.info(f"Successfully generated response for question: {question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            raise
    
    def generate_insights(self, df: pd.DataFrame, data_summary: Dict[str, Any]) -> str:
        """
        Generate automatic insights about the dataset
        
        Args:
            df: DataFrame to analyze
            data_summary: Summary statistics
            
        Returns:
            Generated insights
        """
        prompt = f"""Based on this dataset summary, provide 5-7 key insights about the data:

{json.dumps(data_summary, indent=2)}

Focus on:
1. Notable patterns or trends
2. Data quality issues (missing values, outliers)
3. Relationships between columns
4. Interesting statistics
5. Suggestions for further analysis

Format the insights as a numbered list."""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analyst providing insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            insights = response.choices[0].message.content
            logger.info("Successfully generated insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            raise
    
    def suggest_visualizations(self, df: pd.DataFrame, data_summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Suggest appropriate visualizations for the dataset
        
        Args:
            df: DataFrame to analyze
            data_summary: Summary statistics
            
        Returns:
            List of visualization suggestions
        """
        prompt = f"""Based on this dataset, suggest 3-5 effective visualizations:

Columns: {data_summary['columns']}
Data types: {json.dumps(data_summary['dtypes'], indent=2)}

For each visualization, provide:
1. Type (e.g., bar chart, scatter plot, line chart, histogram)
2. Which columns to use
3. What insight it would reveal

Return as a JSON array with objects containing: type, columns, insight"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data visualization expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            # Parse JSON response
            suggestions_text = response.choices[0].message.content
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in suggestions_text:
                suggestions_text = suggestions_text.split("```json")[1].split("```")[0].strip()
            elif "```" in suggestions_text:
                suggestions_text = suggestions_text.split("```")[1].split("```")[0].strip()
            
            suggestions = json.loads(suggestions_text)
            logger.info(f"Generated {len(suggestions)} visualization suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating visualization suggestions: {str(e)}")
            # Return empty list on error
            return []
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.conversation_history.copy()