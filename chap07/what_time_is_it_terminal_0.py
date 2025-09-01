from gpt_functions_0 import get_current_time, tools
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# 터미널 대화 예제

# 사용자  :뉴욕, 런던, 파리 시간 알려줘
# ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_U21zsZz8qojBd67Ez9aIO9Q3', function=Function(arguments='{"timezone": "America/New_York"}', name='get_current_time'), type='function'), ChatCompletionMessageToolCall(id='call_krte50s6dcnGUjrMeORTKvry', function=Function(arguments='{"timezone": "Europe/London"}', name='get_current_time'), type='function'), ChatCompletionMessageToolCall(id='call_Qr0D53ZC0QScMV7S8TbV5AKX', function=Function(arguments='{"timezone": "Europe/Paris"}', name='get_current_time'), type='function')], annotations=[])
# 2025-08-31 22:45:23 America/New_York
# 2025-09-01 03:45:23 Europe/London
# 2025-09-01 04:45:23 Europe/Paris
# AI      : 현재 시각은 다음과 같습니다:
# - 뉴욕: 2025년 8월 31일 22시 45분
# - 런던: 2025년 9월 1일 03시 45분
# - 파리: 2025년 9월 1일 04시 45분
# 사용자  :exit

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = api_key)

def get_ai_response(messages, tools=None):
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
        tools = tools,
    )
    return response

messages = [
    {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
]

while True:
    user_input = input("사용자\t:")
    if user_input == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    ai_response = get_ai_response(messages, tools)
    ai_message = ai_response.choices[0].message
    print(ai_message)

    tool_calls = ai_message.tool_calls


    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id
            arguments = json.loads(tool_call.function.arguments)

            if tool_name == "get_current_time":
                messages.append({
                    "role": "function",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": get_current_time(timezone=arguments['timezone']),
                })
        messages.append({"role": "system", "content": "이제 주어진 결과를 바탕으로 답변할 차례다."})
        
        ai_response = get_ai_response(messages, tools=tools)
        ai_message = ai_response.choices[0].message

    messages.append(ai_message)

    print("AI\t: " + ai_message.content)