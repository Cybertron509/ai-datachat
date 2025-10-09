from openai import OpenAI

client = OpenAI(api_key="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
")

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("SUCCESS: API key works!")
except Exception as e:
    print(f"ERROR: {e}")