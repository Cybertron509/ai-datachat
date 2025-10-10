"""
AI Agent - Azure OpenAI Only
Copyright © 2025 Gardel Hiram. All rights reserved.
"""
import os
import logging
from typing import Optional
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

RECOMMENDED_API_VERSION = "2024-06-01"

def _clean_endpoint(url: Optional[str]) -> Optional[str]:
    return url.rstrip("/") if url else url

class AIAgent:
    def __init__(self):
        """Initialize AI agent with Azure OpenAI"""

        # --- Env vars (support both your names and standard ones) ---
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", RECOMMENDED_API_VERSION)
        endpoint = _clean_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT"))
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        allow_cogsvc = os.getenv("AZURE_OPENAI_ALLOW_COGSERVICES", "false").lower() in {"1","true","yes"}

        # --- Diagnostics (never log secrets) ---
        logger.info("🔍 Azure OpenAI config:")
        logger.info(f"• API key present: {bool(api_key)}")
        logger.info(f"• Endpoint: {endpoint}")
        logger.info(f"• Deployment: {deployment}")
        logger.info(f"• API version: {api_version}")
        logger.info(f"• Allow Cognitive Services endpoint: {allow_cogsvc}")

        # --- Validate ---
        if not api_key:
            raise ValueError("Missing API key: set AZURE_OPENAI_API_KEY (or AZURE_OPENAI_KEY).")
        if not endpoint:
            raise ValueError("Missing endpoint: set AZURE_OPENAI_ENDPOINT.")
        if not deployment:
            raise ValueError("Missing deployment name: set AZURE_OPENAI_DEPLOYMENT to your exact deployment.")

        # Default: require *.openai.azure.com (Azure OpenAI resource).
        # Only allow *.cognitiveservices.azure.com if explicitly opted in.
        if "openai.azure.com" not in endpoint:
            if "cognitiveservices.azure.com" in endpoint and allow_cogsvc:
                logger.warning(
                    "Using a Cognitive Services endpoint. Ensure your key/endpoint belong to the same multi-service resource."
                )
            else:
                raise ValueError(
                    "Endpoint must be your Azure OpenAI endpoint like "
                    "'https://<resource>.openai.azure.com'. "
                    "If you intentionally use a Cognitive Services endpoint, set AZURE_OPENAI_ALLOW_COGSERVICES=true."
                )

        # --- Init client ---
        try:
            logger.info(f"🔧 Initializing Azure OpenAI client for deployment '{deployment}'")
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
            )
            self.model = deployment
            logger.info("✅ Azure OpenAI initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI: {e}")
            raise

    def health_check(self) -> str:
        """Quick ping to verify key/endpoint/deployment/api-version alignment."""
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                temperature=0.0,
                max_tokens=5,
            )
            txt = (r.choices[0].message.content or "").strip()
            return f"Azure OpenAI OK (reply: {txt[:24]!r})"
        except Exception as e:
            msg = (
                f"Azure OpenAI health_check failed: {e}\n\n"
                "Checklist:\n"
                "  • Key belongs to the same resource as the endpoint\n"
                "  • Endpoint is https://ai-datachat-openai.openai.azure.com\n"
                "  • Deployment name matches the one in Azure → Deployments\n"
                f"  • API version supported in region (e.g., {RECOMMENDED_API_VERSION})\n"
            )
            logger.error(msg)
            raise

    def analyze_data(self, df, query, data_summary=None):
        """
        Analyze data and respond to user queries.

        Args:
            df: pandas DataFrame
            query: user's question
            data_summary: optional summary statistics

        Returns:
            str: AI response
        """
        try:
            context = self._prepare_context(df, data_summary)

            system_prompt = (
                "You are an expert data analyst assistant. "
                "Analyze the provided data and answer questions clearly and concisely. "
                "Provide actionable insights when relevant."
            )

            user_prompt = (
                f"Data Context:\n{context}\n\n"
                f"User Question: {query}\n\n"
                "Please provide a clear, helpful answer based on the data."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"❌ Error in AI analysis: {e}")
            return (
                "Sorry, I encountered an error while contacting Azure OpenAI. "
                "Please verify your AZURE_OPENAI_* settings (key/endpoint/deployment/api-version) and try again.\n\n"
                f"Details: {e}"
            )

    def _prepare_context(self, df, data_summary):
        """Prepare data context for AI"""
        context = (
            f"\nDataset Overview:\n"
            f"- Rows: {len(df):,}\n"
            f"- Columns: {len(df.columns)}\n"
            f"- Column Names: {', '.join(df.columns.tolist()[:20])}\n"
        )
        if data_summary:
            context += f"\nData Summary:\n{str(data_summary)[:500]}"
        context += f"\n\nSample Data (first 3 rows):\n{df.head(3).to_string()}"
        return context
