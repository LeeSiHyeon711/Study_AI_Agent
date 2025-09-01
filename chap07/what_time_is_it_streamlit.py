from gpt_functions_0 import get_current_time, tools
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import streamlit as st

# 실행 방법 : streamlit run chap07/what_time_is_it_streamlit.py

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

st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
    ]

# 이전 메시지들 표시
for msg in st.session_state.messages:
    if msg["role"] != "system":  # 시스템 메시지는 표시하지 않음
        st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력 받기
if user_input := st.chat_input("메시지를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # AI 응답 생성
    ai_response = get_ai_response(st.session_state.messages, tools)
    ai_message = ai_response.choices[0].message
    tool_calls = ai_message.tool_calls
    
    # Assistant 메시지를 먼저 세션에 추가 (tool_calls가 있더라도)
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_message.content,
        "tool_calls": tool_calls
    })
    
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id
            arguments = json.loads(tool_call.function.arguments)

            if tool_name == "get_current_time":
                tool_result = get_current_time(timezone=arguments['timezone'])
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result,
                })
        
        # 툴 실행 후 최종 응답 생성
        final_response = get_ai_response(st.session_state.messages)
        final_message = final_response.choices[0].message
        
        # 최종 응답을 세션에 추가
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_message.content,
        })
        
        # 최종 응답 표시
        st.chat_message("assistant").write(final_message.content)
    else:
        # 툴 호출이 없는 경우 바로 응답 표시
        st.chat_message("assistant").write(ai_message.content)