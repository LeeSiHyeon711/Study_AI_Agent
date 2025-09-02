import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
import os

from langchain_core.tools import tool
from datetime import datetime
import pytz

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# 도구 함수 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수"""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재 시각 {now}'
        print(result)
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"

@tool
def get_web_search(query: str, search_period: str) -> str:

    """
    웹 검색을 수행하는 함수

    Args:
        query (str): 검색어
        search_period (str): 검색 기간(e.g., "w" for past week, "m" for past month, "y" for past year)
    
    Returns:
        str : 검색 결과
    """
    wrapper = DuckDuckGoSearchAPIWrapper(region="kr-ko", time=search_period)

    print('------------ WEB SEARCH ------------')
    print(query)
    print(search_period)

    search = DuckDuckGoSearchResults(
        api_wrapper = wrapper,
        results_separator = "\n",
    )

    docs = search.invoke(query)
    return docs

@tool
def get_youtube_search(query: str) -> str:
    """
    유튜브 검색을 한 뒤, 영상들의 내용을 반환하는 함수.

    Args:
        query (str): 검색어
    
    Returns:
        str: 검색 결과
    """
    print('------------ YOUTUBE SEARCH ------------')
    print(query)

    videos = YoutubeSearch(query, max_results=5).to_dict()

    videos = [video for video in videos if len(video['duration']) <= 5]

    for video in videos:
        video_url = 'https://youtube.com' + video['url_suffix']
        video_id = video['url_suffix'].split('v=')[-1] if 'v=' in video['url_suffix'] else video['url_suffix'].split('/')[-1]

        try:
            # 자막 가져오기
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
            transcript_text = ' '.join([item['text'] for item in transcript_list])
            
            video['video_url'] = video_url
            video['content'] = transcript_text
        except Exception as e:
            print(f"자막을 가져올 수 없습니다: {e}")
            video['video_url'] = video_url
            video['content'] = "자막을 가져올 수 없습니다."
    
    return str(videos)

# 도구 바인딩
tools = [get_current_time, get_web_search, get_youtube_search]
tool_dict = {
    "get_current_time": get_current_time,
    "get_web_search": get_web_search,
    "get_youtube_search": get_youtube_search,
}

llm_with_tools = llm.bind_tools(tools)

# 사용자의 메시지를 처리하는 함수
def get_ai_response(messages):
    response = llm_with_tools.stream(messages)

    gathered = None
    for chunk in response:
        yield chunk

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk
    
    if gathered.tool_calls:
        st.session_state.messages.append(gathered)

        for tool_call in gathered.tool_calls:
            selected_tool = tool_dict[tool_call['name']]
            tool_result = selected_tool.invoke(tool_call['args'])
            print(tool_result, type(tool_result))
            
            # ToolMessage 생성
            tool_msg = ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call['id']
            )
            st.session_state.messages.append(tool_msg)
        
        for chunk in get_ai_response(st.session_state.messages):
            yield chunk

# 스트림릿 앱
st.title("GPT-4o Langchain Chat")

# 스트림릿 session_state에 메시지 저장
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다."),
    ]

# 스트림릿 화면에 메시지 출력
for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, SystemMessage):
            st.chat_message("system").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, ToolMessage):
            st.chat_message("tool").write(msg.content)

# 사용자 입력 처리
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    response = get_ai_response(st.session_state["messages"])

    result = st.chat_message("assistant").write_stream(response)
    st.session_state["messages"].append(AIMessage(result))