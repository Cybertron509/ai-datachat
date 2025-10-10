"""
AI Agent - Azure OpenAI Only
Copyright © 2025 Gardel Hiram. All rights reserved.
"""
import os
from openai import AzureOpenAI
import logging

logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self):
        """Initialize AI agent with Azure OpenAI"""
        try:
            # Get configuration from environment variables - matching YOUR Render variable names
            api_key = os.getenv('AZURE_OPENAI_KEY')
            api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-35-turbo')
            
            # DEBUG: Print what we found
            logger.info(f"🔍 API Key exists: {bool(api_key)}")
            logger.info(f"🔍 Endpoint: {endpoint}")
            logger.info(f"🔍 Deployment: {deployment}")
            logger.info(f"🔍 API Version: {api_version}")
            
            # Validate required variables
            if not api_key:
                raise ValueError("AZURE_OPENAI_KEY is not set")
            if not endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
            if not deployment:
                raise ValueError("AZURE_OPENAI_DEPLOYMENT is not set")
            
            logger.info(f"🔧 Initializing Azure OpenAI with deployment: {deployment}")
            
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            self.model = deployment
            
            logger.info("✅ Azure OpenAI initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI: {str(e)}")
            raise
    
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
            
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ Error in AI analysis: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}. Please check your Azure OpenAI configuration."
    
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