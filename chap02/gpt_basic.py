from openai import OpenAI
from dotenv import load_dotenv
import os

# 단일 대화 예제

# ChatCompletion(id='chatcmpl-CAibDz0BG0FCJ2S6MgI6YjIXV8u7d', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='2022년 FIFA 월드컵에 
# 서는 아르헨티나가 우승을 차지했습니다. 아르헨티나는 결승전에서 프랑스를 상대로 승리하여 월 
# 드컵 트로피를 들어올렸습니다.', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None, annotations=[]))], created=1756670755, model='gpt-4o-2024-08-06', object='chat.completion', service_tier='default', system_fingerprint='fp_80956533cb', usage=CompletionUsage(completion_tokens=52, prompt_tokens=30, total_tokens=82, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
# ----
# 2022년 FIFA 월드컵에서는 아르헨티나가 우승을 차지했습니다. 아르헨티나는 결승전에서 프랑스를
#  상대로 승리하여 월드컵 트로피를 들어올렸습니다.

load_dotenv()  # .env 파일에서 환경 변수 로드

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model = "gpt-4o",
    temperature=0.1,
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "2022년 월드컵 우승 팀은 어디야?"},
    ]
)

print(response)

print('----')
print(response.choices[0].message.content)