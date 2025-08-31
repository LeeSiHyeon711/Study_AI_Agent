from openai import OpenAI
from dotenv import load_dotenv
import os

# 싱글턴 대화 예제

# 사용자: 안녕 내 이름은 이시현이야
# AI: 안녕하세요, 시현님! 만나서 반갑습니다. 오늘 어떻게 도와드릴까요?
# 사용자: 내 이름이 뭘까?
# AI: 죄송하지만, 당신의 이름을 모릅니다. 당신의 이름을 알려주시면 기억하는 데 도움
# 이 되겠습니다.
# 사용자: exit

load_dotenv()  # .env 파일에서 환경 변수 로드
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = api_key)

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    response = client.chat.completions.create(
        model = "gpt-4o",
        temperature=0.9,
        messages = [
            {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
            {"role": "user", "content": user_input},
        ],
    )
    print("AI: " + response.choices[0].message.content)