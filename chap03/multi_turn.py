from openai import OpenAI
from dotenv import load_dotenv
import os

# 멀티턴 대화 예제

# 사용자: 안녕 내 이름은 이시현이야
# AI: 안녕하세요, 시현님! 만나서 반갑습니다. 오늘 어떻게 도와드릴까요?
# 사용자: 내가 누구게?
# AI: 시현님이라고 소개해 주셨는데, 더 구체적으로 말씀해주시면 더 잘 이해할 수 있을 
# 것 같아요. 혹시나 특별한 이야기를 나누고 싶으시다면 언제든지 말씀해 주세요!       
# 사용자: exit

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = api_key)

def get_ai_response(messages):
    response = client.chat.completions.create(
        model = "gpt-4o",       # 응답 생성에 사용할 모델 지정
        temperature = 0.9,      # 응답의 창의성 조절
        messages = messages,    # 대화 기록을 입력으로 전달
    )
    return response.choices[0].message.content  # AI의 응답 반환

messages = [
    {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
]

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    messages.append({"role": "user", "content": user_input})  # 사용자 입력 추가
    ai_response = get_ai_response(messages)  # AI 응답 생성
    messages.append({"role": "assistant", "content": ai_response})
    print("AI: " + ai_response) # AI 응답 출력