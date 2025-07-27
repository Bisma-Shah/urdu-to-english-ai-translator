from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()  # Load env variables

gemini_api_key = os.getenv("GEMINI-API-KEY")  # Get API key

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is missing in .env")  # Error if key missing

# Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Load Gemini model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

# Configure run settings
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# Define translator agent
translator = Agent(
    name="translator",
    instructions="Translate Urdu to English accurately. Just give the English translation—no explanations.",
)

# Run agent with input
response = Runner.run_sync(
    translator,
    input="خواب وہ نہیں جو نیند میں آئیں، خواب وہ ہیں جو سونے نہ دیں",
    run_config=config,
)

print(response.final_output)  # Show result
