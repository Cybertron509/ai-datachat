"""AI Agent for data analysis using OpenAI"""
from openai import OpenAI
import pandas as pd
import json
from typing import Dict, Any, List, Optional
import logging
from config import settings

logger = logging.getLogger(__name__)

class AIAgent:
    """AI agent for analyzing data and answering questions"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize AI Agent"""
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"Initialized AI Agent with model: {self.model}")
    
    def _create_system_prompt(self, df: pd.DataFrame, data_summary: Dict[str, Any]) -> str:
        """Create system prompt with data context"""
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

You have access to the complete dataset and can reference any columns or statistics about the data.
"""
        return prompt
    
    def analyze_data(self, df: pd.DataFrame, question: str, data_summary: Dict[str, Any]) -> str:
        """Analyze data and answer a question"""
        try:
            system_prompt = self._create_system_prompt(df, data_summary)
            data_sample = df.head(5).to_string()
            context_message = f"Here's a sample of the data:\n{data_sample}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_message}
            ]
            
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": question})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            logger.info(f"Successfully generated response for question: {question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            raise
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")