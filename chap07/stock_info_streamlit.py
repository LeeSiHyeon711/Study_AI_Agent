from gpt_functions_0 import get_current_time, tools, get_yf_stock_info, get_yf_stock_history, get_yf_stock_recommendations
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import streamlit as st
from collections import defaultdict

# 실행 방법 : streamlit run chap07/what_time_is_it_streamlit.py

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = api_key)

def tool_list_to_tool_obj(tools):
    tool_calls_dict = defaultdict(lambda: {"id": None, "function": {"arguments": "", "name": None}, "type": None})

    for tool_call in tools:
        # id 가 None이 아닌 경우 설정
        if tool_call.id is not None:
            tool_calls_dict[tool_call.index]["id"] = tool_call.id
        
        # 함수 이름이 None이 아닌 경우 설정
        if tool_call.function.name is not None:
            tool_calls_dict[tool_call.index]["function"]["name"] = tool_call.function.name

        # 인자 추가
        tool_calls_dict[tool_call.index]["function"]["arguments"] += tool_call.function.arguments

        # 타입이 None이 아닌 경우 설정
        if tool_call.type is not None:
            tool_calls_dict[tool_call.index]["type"] = tool_call.type
        
    tool_calls_list = list(tool_calls_dict.values())

    return {"tool_calls": tool_calls_list}

def get_ai_response(messages, tools=None, stream=True):
    response = client.chat.completions.create(
        model = "gpt-4o",
        stream = stream,
        messages = messages,
        tools = tools,
    )

    if stream:
        for chunk in response:
            yield chunk
    else:
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
    content = ''
    tool_calls_chunk = []

    with st.chat_message("assistant"):
        placeholder = st.empty()
        for chunk in ai_response:
            content_chunk = chunk.choices[0].delta.content
            if content_chunk:
                print(content_chunk, end="")
                content += content_chunk
                placeholder.markdown(content)
            if chunk.choices[0].delta.tool_calls:
                tool_calls_chunk += chunk.choices[0].delta.tool_calls
        
        tool_obj = tool_list_to_tool_obj(tool_calls_chunk)
        tool_calls = tool_obj["tool_calls"]

        if len(tool_calls) > 0:
            print(tool_calls)
            tool_call_msg = [tool_call["function"] for tool_call in tool_calls]
            st.write(tool_call_msg)
            
    print('\n==============')
    print(content)

    # print('\n============== tool_calls_chunk ==============')
    # for tool_call_chunk in tool_calls_chunk:
    #     print(tool_call_chunk)
    # ai_message = ai_response.choices[0].message
    # tool_calls = ai_message.tool_calls
    tool_obj = tool_list_to_tool_obj(tool_calls_chunk)
    tool_calls = tool_obj["tool_calls"]
    print(tool_calls)
    
    if tool_calls:
        # 1. tool_calls가 포함된 어시스턴트 메시지를 먼저 추가
        api_tool_calls = []
        for tool_call in tool_calls:
            api_tool_calls.append({
                "id": tool_call["id"],
                "type": tool_call["type"],
                "function": {
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"]
                }
            })
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": api_tool_calls
        })
        
        # 2. 툴들을 실행하고 결과를 추가
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_call_id = tool_call["id"]
            arguments = json.loads(tool_call["function"]["arguments"])

            if tool_name == "get_current_time":
                tool_result = get_current_time(timezone=arguments['timezone'])
            elif tool_name == "get_yf_stock_info":
                tool_result = get_yf_stock_info(ticker=arguments['ticker'])
            elif tool_name == "get_yf_stock_history":
                tool_result = get_yf_stock_history(ticker=arguments['ticker'], period=arguments['period'])
            elif tool_name == "get_yf_stock_recommendations":
                tool_result = get_yf_stock_recommendations(ticker=arguments['ticker'])

            st.session_state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": tool_result,
            })
        
        # 3. 툴 실행 후 최종 응답 생성 (스트리밍)
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.messages,
            tools=tools,
            stream=True
        )
        
        # 4. 최종 응답을 스트리밍으로 표시
        final_content = ''
        with st.chat_message("assistant"):
            final_placeholder = st.empty()
            for chunk in final_response:
                final_content_chunk = chunk.choices[0].delta.content
                if final_content_chunk:
                    print(final_content_chunk, end="")
                    final_content += final_content_chunk
                    final_placeholder.markdown(final_content)
        
        # 5. 최종 응답을 세션에 추가
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_content,
        })
    else:
        # 툴 호출이 없는 경우 응답을 세션에 추가
        st.session_state.messages.append({
            "role": "assistant",
            "content": content,
        })