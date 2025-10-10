"""
AI Agent - Azure OpenAI Compatible
"""
import os
from openai import AzureOpenAI, OpenAI
import logging

logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        """Initialize AI agent with Azure OpenAI or regular OpenAI"""
        self.use_azure = os.getenv('USE_AZURE_OPENAI', 'false').lower() == 'true'
        
        if self.use_azure:
            # Use Azure OpenAI
            self.client = AzureOpenAI(
                api_key=os.getenv('AZURE_OPENAI_KEY'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
            )
            self.model = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-35-turbo')
            logger.info("✅ Using Azure OpenAI")
        else:
            # Use regular OpenAI
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model = "gpt-3.5-turbo"
            logger.info("✅ Using OpenAI Direct")
    
    def analyze_data(self, df, query, data_summary=None):
        """
        Analyze data and respond to user queries
        
        Args:
            df: pandas DataFrame
            query: user's question
            data_summary: optional summary statistics
        
        Returns:
            str: AI response
        """
        try:
            # Prepare context about the data
            context = self._prepare_context(df, data_summary)
            
            # Create the prompt
            system_prompt = """You are an expert data analyst assistant. 
            Analyze the provided data and answer questions clearly and concisely.
            Provide actionable insights when relevant."""
            
            user_prompt = f"""Data Context:
{context}

User Question: {query}

Please provide a clear, helpful answer based on the data."""
            
            # Call Azure OpenAI or regular OpenAI (same API!)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _prepare_context(self, df, data_summary):
        """Prepare data context for AI"""
        context = f"""
Dataset Overview:
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Column Names: {', '.join(df.columns.tolist()[:20])}
"""
        
        if data_summary:
            context += f"\nData Summary:\n{str(data_summary)[:500]}"
        
        # Add sample data
        context += f"\n\nSample Data (first 3 rows):\n{df.head(3).to_string()}"
        
        return context