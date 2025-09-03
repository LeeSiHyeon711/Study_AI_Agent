# Study AI Agent

AI Agent 학습을 위한 실습 프로젝트입니다. OpenAI GPT API를 활용한 다양한 대화형 AI 애플리케이션을 단계별로 구현해보며 AI Agent의 기본 개념과 활용법을 학습합니다.

## 📁 프로젝트 구조

```
Study_AI_Agent/
├── chap02/                 # 2장: GPT API 기본 사용법
│   └── gpt_basic.py       # 단일 대화 예제
├── chap03/                 # 3장: 대화형 AI 애플리케이션
│   ├── single_turn.py     # 싱글턴 대화 (메모리 없음)
│   ├── multi_turn.py      # 멀티턴 대화 (메모리 있음)
│   └── streamlit_basic.py # Streamlit 웹 인터페이스
├── chap07/                 # 7장: GPT Functions (함수 호출)
│   ├── gpt_functions_0.py # 시간 조회 및 주식 정보 함수 정의
│   ├── what_time_is_it_terminal_0.py # 터미널 기반 함수 호출 예제
│   ├── what_time_is_it_streamlit.py # Streamlit 기반 시간 조회 예제
│   └── stock_info_streamlit.py # Streamlit 기반 주식 정보 조회 예제
├── chap08/                 # 8장: LangChain Tools
│   ├── langchain_tool.ipynb # LangChain을 활용한 도구 연동 예제
│   ├── langchain_simple_chat_streamlit.py # LangChain 기본 채팅 (메시지 히스토리)
│   ├── langchain_streamlit_tool_0.py # LangChain 스트리밍 채팅 (기본)
│   └── langchain_streamlit_tool.py # LangChain 도구 연동 스트리밍 채팅
├── chap09/                 # 9장: RAG (Retrieval-Augmented Generation)
│   ├── rag_practice.ipynb # RAG 시스템 구현 실습
│   ├── rag.py             # Streamlit 기반 RAG 채팅 애플리케이션
│   ├── retriever.py       # RAG 검색 및 문서 처리 모듈
│   └── data/              # RAG 학습용 데이터
│       ├── OneNYC-2050-Summary.pdf # 뉴욕시 전략 계획서 요약본
│       └── 2040_seoul_plan.pdf # 서울시 2040 계획서
├── chap10/                 # 10장: 웹 검색 및 유튜브 연동
│   ├── streamlit_with_web_search.py # 웹 검색, 유튜브 검색, 시간 조회 통합 도구
│   ├── duckduckgo.ipynb   # DuckDuckGo 웹 검색 도구 실습
│   ├── youtube_summary.ipynb # 유튜브 동영상 요약 실습
│   └── youtube_summary_fixed.py # 유튜브 자막 추출 및 처리 모듈
├── chap11/                 # 11장: 로컬 LLM (DeepSeek)
│   └── deepseek_simple_chatbot.py # Ollama 기반 DeepSeek LLM 채팅봇
├── chap12/                 # 12장: LangGraph 기반 AI Agent
│   ├── langgraph_memory.py # LangGraph를 활용한 메모리 기반 대화 시스템
│   ├── langgraph_simple_chatbot.ipynb # LangGraph 기본 채팅봇 구현
│   └── langgraph_tools.ipynb # LangGraph 도구 연동 시스템
├── chap13/                 # 13장: LangGraph + RAG 통합 시스템
│   ├── rag_with_langgraph.ipynb # LangGraph와 RAG를 결합한 고급 AI Agent
│   ├── chroma_store/      # 벡터 데이터베이스 저장소
│   └── data/              # RAG 학습용 데이터
│       ├── OneNYC-2050-Summary.pdf # 뉴욕시 전략 계획서 요약본
│       └── 2040_seoul_plan.pdf # 서울시 2040 계획서
├── chap14/                 # 14장: 다중 AI Agent 협업 시스템
│   ├── book_writer.py     # 다중 AI Agent를 활용한 책 작성 시스템
│   ├── models.py          # Task 모델 정의
│   ├── utils.py           # 상태 저장 및 목차 관리 유틸리티
│   ├── book_writer.png    # LangGraph 시각화 이미지
│   └── data/              # 작업 상태 및 목차 데이터
│       ├── state.json     # AI Agent 작업 상태 저장
│       └── outline.md     # 생성된 책 목차
├── AI/                     # 가상환경 및 패키지
├── .venv/                  # 가상환경
├── .gitignore             # Git 무시 파일 설정
└── README.md              # 프로젝트 설명서
```

## 🚀 시작하기

### 1. 환경 설정

1. **가상환경 활성화**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **필요한 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```
   
   또는 개별 설치:
   ```bash
   pip install openai>=1.0.0 python-dotenv>=1.0.0 streamlit>=1.28.0 pytz>=2023.3 yfinance>=0.2.0 langchain>=0.3.0 langchain-openai>=0.3.0 langchain-core>=0.3.0 langchain-community>=0.3.0 langchain-chroma>=0.3.0 langchain-text-splitters>=0.3.0 chromadb>=0.4.0 tabulate>=0.9.0
   ```

3. **환경 변수 설정**
   - `.env` 파일을 생성하고 OpenAI API 키를 설정하세요:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### 2. OpenAI API 키 발급

1. [OpenAI Platform](https://platform.openai.com/account/api-keys)에 접속
2. 계정 생성 또는 로그인
3. API 키 생성 및 복사
4. `.env` 파일에 API 키 추가

## 📚 학습 내용

### Chapter 2: GPT API 기본 사용법

**파일**: `chap02/gpt_basic.py`

- OpenAI GPT API 기본 연결 방법
- 단일 대화 요청 및 응답 처리
- 시스템 메시지와 사용자 메시지 구조 이해
- API 응답 객체 구조 파악

**실행 방법**:
```bash
python chap02/gpt_basic.py
```

### Chapter 3: 대화형 AI 애플리케이션

#### 3.1 싱글턴 대화 (`single_turn.py`)

- 메모리가 없는 단순한 대화 시스템
- 각 대화가 독립적으로 처리됨
- 이전 대화 내용을 기억하지 못함

**실행 방법**:
```bash
python chap03/single_turn.py
```

#### 3.2 멀티턴 대화 (`multi_turn.py`)

- 대화 기록을 유지하는 시스템
- 이전 대화 내용을 기억하고 맥락을 이해
- `messages` 리스트를 통해 대화 히스토리 관리

**실행 방법**:
```bash
python chap03/multi_turn.py
```

#### 3.3 Streamlit 웹 인터페이스 (`streamlit_basic.py`)

- 웹 기반 채팅 인터페이스
- Streamlit을 활용한 사용자 친화적 UI
- 실시간 대화 가능

**실행 방법**:
```bash
streamlit run chap03/streamlit_basic.py
```

### Chapter 7: GPT Functions (함수 호출)

#### 7.1 함수 정의 (`gpt_functions_0.py`)

- GPT가 호출할 수 있는 함수 정의
- 시간 조회 및 주식 정보 조회 함수 구현
- 함수 스키마 정의 및 파라미터 설정

**주요 기능**:
- **시간 조회**: 다양한 타임존의 현재 시간 조회 (`get_current_time`)
- **주식 정보**: Yahoo Finance API를 통한 주식 데이터 조회
  - `get_yf_stock_info`: 종목 기본 정보
  - `get_yf_stock_history`: 주가 히스토리 데이터
  - `get_yf_stock_recommendations`: 투자 추천 정보
- `pytz` 및 `yfinance` 라이브러리 활용

#### 7.2 터미널 기반 함수 호출 (`what_time_is_it_terminal_0.py`)

- 콘솔 환경에서 GPT Functions 사용
- 함수 호출 및 결과 처리 로직
- 다중 함수 호출 지원

**실행 방법**:
```bash
python chap07/what_time_is_it_terminal_0.py
```

**사용 예시**:
```
사용자: 뉴욕, 런던, 파리 시간 알려줘
AI: 현재 시각은 다음과 같습니다:
- 뉴욕: 2025년 8월 31일 22시 45분
- 런던: 2025년 9월 1일 03시 45분
- 파리: 2025년 9월 1일 04시 45분
```

#### 7.3 Streamlit 기반 시간 조회 (`what_time_is_it_streamlit.py`)

- 웹 인터페이스에서 시간 조회 함수 활용
- 실시간 함수 호출 및 결과 표시
- 사용자 친화적인 채팅 인터페이스

**실행 방법**:
```bash
streamlit run chap07/what_time_is_it_streamlit.py
```

#### 7.4 Streamlit 기반 주식 정보 조회 (`stock_info_streamlit.py`)

- 웹 인터페이스에서 주식 정보 조회 함수 활용
- **스트리밍 응답**: 실시간 타이핑 효과로 응답 표시
- **다중 함수 호출**: 시간 조회와 주식 정보를 동시에 조회 가능
- **고급 메시지 처리**: OpenAI API 메시지 순서 최적화

**주요 특징**:
- 스트리밍 방식의 실시간 응답 표시
- 툴 호출 후 최종 응답도 스트리밍으로 처리
- 메시지 순서 최적화로 안정적인 함수 호출

**실행 방법**:
```bash
streamlit run chap07/stock_info_streamlit.py
```

### Chapter 8: LangChain Tools

#### 8.1 Jupyter 노트북 기반 도구 연동 (`langchain_tool.ipynb`)

- **LangChain 프레임워크**: OpenAI API를 더 체계적으로 활용하는 프레임워크
- **Tool 시스템**: LangChain의 `@tool` 데코레이터를 사용한 함수 정의
- **Pydantic 모델**: 타입 안전성을 위한 입력값 검증 및 구조화
- **Jupyter 노트북**: 인터랙티브한 학습 환경

**주요 기능**:
- **시간 조회 도구**: `get_current_time` 함수로 다양한 타임존의 현재 시간 조회
- **주식 데이터 조회**: `get_yf_stock_history` 함수로 Yahoo Finance API 연동
- **도구 바인딩**: `llm.bind_tools()`를 통한 모델과 도구의 연결
- **메시지 처리**: LangChain의 메시지 시스템을 통한 대화 관리

**실행 방법**:
```bash
jupyter notebook chap08/langchain_tool.ipynb
```

#### 8.2 LangChain 기본 채팅 (`langchain_simple_chat_streamlit.py`)

- **메시지 히스토리 관리**: `RunnableWithMessageHistory`를 통한 대화 기록 유지
- **세션 관리**: 사용자별 독립적인 대화 세션 지원
- **스트리밍 응답**: 실시간 타이핑 효과로 응답 표시
- **LangChain 메시지 시스템**: `SystemMessage`, `HumanMessage`, `AIMessage` 활용

**주요 특징**:
- **세션 기반 대화**: 각 사용자별로 독립적인 대화 히스토리 관리
- **메모리 유지**: 이전 대화 내용을 기억하고 맥락을 이해
- **실시간 스트리밍**: 응답이 실시간으로 타이핑되듯이 표시

**실행 방법**:
```bash
streamlit run chap08/langchain_simple_chat_streamlit.py
```

#### 8.3 LangChain 스트리밍 채팅 (`langchain_streamlit_tool_0.py`)

- **기본 스트리밍**: LangChain의 `stream()` 메서드를 활용한 실시간 응답
- **메시지 관리**: Streamlit session_state를 통한 대화 기록 저장
- **간단한 구조**: 도구 없이 순수한 대화 기능에 집중

**주요 특징**:
- **스트리밍 응답**: `llm.stream()`을 통한 실시간 응답 처리
- **메시지 타입 구분**: 시스템, 사용자, AI 메시지를 구분하여 표시
- **세션 상태 관리**: Streamlit의 session_state를 활용한 대화 기록 유지

**실행 방법**:
```bash
streamlit run chap08/langchain_streamlit_tool_0.py
```

#### 8.4 LangChain 도구 연동 스트리밍 채팅 (`langchain_streamlit_tool.py`)

- **도구 연동**: `@tool` 데코레이터를 사용한 함수 정의 및 호출
- **스트리밍 + 도구**: 스트리밍 응답과 도구 호출의 조합
- **재귀적 처리**: 도구 호출 후 최종 응답도 스트리밍으로 처리
- **시간 조회 기능**: 다양한 타임존의 현재 시간 조회

**주요 특징**:
- **도구 바인딩**: `llm.bind_tools()`를 통한 모델과 도구의 연결
- **재귀적 스트리밍**: 도구 호출 후 최종 응답도 스트리밍으로 표시
- **실시간 도구 실행**: 사용자 질문에 따라 자동으로 도구 호출
- **메시지 순서 관리**: 도구 호출과 응답의 올바른 순서 보장

**실행 방법**:
```bash
streamlit run chap08/langchain_streamlit_tool.py
```

**사용 예시**:
```python
# 도구 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수"""
    # 구현 내용...

# 모델에 도구 바인딩
llm_with_tools = llm.bind_tools([get_current_time])

# 사용자 질문
messages = [HumanMessage("부산은 지금 몇 시야?")]
response = llm_with_tools.invoke(messages)
```

### Chapter 9: RAG (Retrieval-Augmented Generation)

#### 9.1 RAG 시스템 구현 실습 (`rag_practice.ipynb`)

- **RAG 개념**: 검색 기반 생성 모델을 통한 정확한 정보 제공
- **문서 처리**: PDF 문서 로딩 및 텍스트 분할
- **벡터 데이터베이스**: Chroma를 활용한 임베딩 저장 및 검색
- **검색 및 생성**: 관련 문서 검색 후 AI 응답 생성

**주요 기능**:
- **문서 로더**: `PyPDFLoader`를 사용한 PDF 문서 로딩
- **텍스트 분할**: `RecursiveCharacterTextSplitter`를 통한 청크 분할
- **임베딩**: OpenAI 임베딩 모델을 사용한 벡터화
- **벡터 저장소**: Chroma 벡터 데이터베이스에 임베딩 저장
- **유사도 검색**: 쿼리와 관련된 문서 청크 검색
- **문맥 기반 응답**: 검색된 문서를 바탕으로 정확한 답변 생성

**학습 데이터**:
- **뉴욕시 전략 계획서**: OneNYC 2050 Strategic Plan (117MB)
- **서울시 2040 계획서**: 2040 Seoul Plan (5.6MB)

**주요 특징**:
- **정확한 정보 제공**: 외부 문서를 참조하여 사실에 기반한 답변
- **실시간 검색**: 사용자 질문에 관련된 문서를 실시간으로 검색
- **문맥 이해**: 검색된 문서의 맥락을 고려한 응답 생성
- **확장 가능성**: 다양한 문서 형식과 소스 지원

**실행 방법**:
```bash
jupyter notebook chap09/rag_practice.ipynb
```

**사용 예시**:
```python
# 문서 로딩
loader = PyPDFLoader("data/OneNYC_2050_Strategic_Plan.pdf")
documents = loader.load()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(splits)

# 벡터 저장소 생성
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)

# RAG 체인 생성
llm = ChatOpenAI(model="gpt-4o-mini")
retrieval_chain = create_stuff_documents_chain(llm, prompt)

# 질문에 대한 답변 생성
question = "뉴욕시의 기후 변화 대응 정책은 무엇인가요?"
answer = retrieval_chain.invoke({"context": docs, "question": question})
```

#### 9.2 RAG 검색 및 문서 처리 모듈 (`retriever.py`)

- **모듈화된 RAG 시스템**: 재사용 가능한 검색 및 문서 처리 컴포넌트
- **Chroma 벡터 저장소**: 기존 벡터 데이터베이스 활용
- **쿼리 증강**: 대화 맥락을 고려한 질문 개선
- **문서 체인**: LangChain을 활용한 문서 기반 응답 생성

**주요 기능**:
- **벡터 저장소 연결**: 기존 Chroma 저장소에 연결하여 문서 검색
- **쿼리 증강 체인**: 대화 히스토리를 활용한 질문 개선
- **문서 응답 체인**: 검색된 문서를 바탕으로 AI 응답 생성
- **임베딩 모델**: OpenAI text-embedding-3-large 모델 사용

**주요 특징**:
- **재사용 가능한 모듈**: 다른 애플리케이션에서 쉽게 활용 가능
- **대화 맥락 이해**: 이전 대화 내용을 고려한 질문 개선
- **효율적인 검색**: 상위 3개 관련 문서 검색 (k=3)
- **모듈화된 구조**: 검색, 증강, 응답 생성 기능 분리

#### 9.3 Streamlit 기반 RAG 채팅 애플리케이션 (`rag.py`)

- **웹 기반 RAG 시스템**: Streamlit을 활용한 사용자 친화적 인터페이스
- **실시간 문서 검색**: 사용자 질문에 대한 즉시 문서 검색 및 표시
- **스트리밍 응답**: 실시간으로 AI 응답 생성 및 표시
- **대화 히스토리**: 세션 기반 대화 기록 유지

**주요 기능**:
- **실시간 문서 검색**: 사용자 질문에 관련된 문서를 즉시 검색
- **문서 미리보기**: 검색된 문서를 확장 가능한 패널로 표시
- **스트리밍 응답**: AI 응답을 실시간으로 타이핑 효과로 표시
- **대화 맥락 유지**: 이전 대화 내용을 기억하고 맥락 이해
- **도시 정책 전문가**: 문서 기반 도시 정책 전문가 역할 수행

**주요 특징**:
- **문서 기반 답변**: 검색된 문서를 바탕으로 정확한 정보 제공
- **사용자 친화적 UI**: Streamlit의 채팅 인터페이스 활용
- **실시간 피드백**: 검색 과정과 응답 생성 과정을 실시간으로 표시
- **세션 관리**: Streamlit session_state를 활용한 대화 기록 관리
- **문서 소스 표시**: 답변의 근거가 되는 문서와 페이지 정보 표시

**실행 방법**:
```bash
streamlit run chap09/rag.py
```

**사용 예시**:
```
사용자: 서울시의 2040 계획에서 교통 정책은 어떻게 되어있나요?
AI: 서울시 2040 계획서를 참조하여 교통 정책에 대해 답변드리겠습니다.

서울시 2040 계획의 주요 교통 정책은 다음과 같습니다:

1. **대중교통 중심 도시**: 지하철, 버스 등 대중교통 확충
2. **친환경 교통**: 전기차, 수소차 등 친환경 차량 보급 확대
3. **보행자 중심**: 보행자 우선 도시 조성
4. **스마트 교통**: AI 기반 교통 관리 시스템 구축

[문서: 2040_seoul_plan.pdf, 페이지: 45-52]
```

### Chapter 10: 웹 검색 및 유튜브 연동

#### 10.1 통합 웹 검색 도구 (`streamlit_with_web_search.py`)

- **웹 검색**: DuckDuckGo API를 활용한 실시간 웹 검색
- **유튜브 검색**: 유튜브 동영상 검색 및 자막 추출
- **시간 조회**: 다양한 타임존의 현재 시간 조회
- **통합 도구 시스템**: 여러 도구를 하나의 인터페이스에서 활용

**주요 기능**:
- **DuckDuckGo 웹 검색**: `get_web_search` 함수로 실시간 웹 검색 결과 제공
- **유튜브 검색 및 자막**: `get_youtube_search` 함수로 동영상 검색 및 자막 추출
- **시간 조회**: `get_current_time` 함수로 전 세계 시간대 지원
- **Streamlit 인터페이스**: 웹 기반 사용자 친화적 도구 활용 환경

**주요 특징**:
- **다중 도구 통합**: 웹 검색, 유튜브, 시간 조회를 하나의 시스템에서 처리
- **실시간 검색**: DuckDuckGo API를 통한 최신 웹 정보 검색
- **유튜브 자막 분석**: 동영상 내용을 자막을 통해 텍스트로 분석
- **검색 기간 설정**: 과거 1주일, 1개월, 1년 등 검색 기간 지정 가능

**실행 방법**:
```bash
streamlit run chap10/streamlit_with_web_search.py
```

#### 10.2 DuckDuckGo 웹 검색 실습 (`duckduckgo.ipynb`)

- **Jupyter 노트북**: 인터랙티브한 웹 검색 도구 학습 환경
- **DuckDuckGo API**: 프라이버시 중심의 검색 엔진 활용
- **검색 결과 처리**: 검색 결과의 구조화 및 분석

**주요 기능**:
- **웹 검색 API**: DuckDuckGo 검색 결과 실시간 조회
- **검색 파라미터**: 지역, 시간, 언어 등 검색 조건 설정
- **결과 구조화**: 검색 결과를 체계적으로 정리 및 표시

**실행 방법**:
```bash
jupyter notebook chap10/duckduckgo.ipynb
```

#### 10.3 유튜브 동영상 요약 실습 (`youtube_summary.ipynb`)

- **유튜브 검색**: 키워드 기반 동영상 검색
- **자막 추출**: YouTube Transcript API를 활용한 자막 데이터 수집
- **내용 분석**: 자막을 통한 동영상 내용 이해 및 요약

**주요 기능**:
- **동영상 검색**: `youtube_search` 라이브러리를 활용한 동영상 검색
- **자막 추출**: 한국어/영어 자막 자동 감지 및 추출
- **내용 요약**: 추출된 자막을 바탕으로 동영상 내용 분석

**실행 방법**:
```bash
jupyter notebook chap10/youtube_summary.ipynb
```

#### 10.4 유튜브 자막 처리 모듈 (`youtube_summary_fixed.py`)

- **자막 추출 모듈**: 재사용 가능한 유튜브 자막 처리 컴포넌트
- **URL 파싱**: 다양한 유튜브 URL 형식에서 video ID 추출
- **다국어 지원**: 한국어 우선, 영어 대체 자막 처리

**주요 기능**:
- **URL 파싱**: 정규표현식을 활용한 video ID 추출
- **자막 가져오기**: YouTube Transcript API를 통한 자막 수집
- **언어 우선순위**: 한국어 자막 우선, 실패 시 영어 자막 시도
- **에러 핸들링**: 자막 추출 실패 시 대체 방안 제공

**사용 예시**:
```python
# 키워드 검색
videos = YoutubeSearch("미국 대선", max_results=5).to_dict()

# 자막 가져오기
transcript = get_youtube_transcript(video_url)

# 전체 자막 텍스트 합치기
full_text = " ".join([segment.text for segment in transcript])
```

### Chapter 11: 로컬 LLM (DeepSeek)

#### 11.1 DeepSeek LLM 채팅봇 (`deepseek_simple_chatbot.py`)

- **로컬 LLM**: Ollama를 통한 로컬에서 실행되는 언어 모델
- **DeepSeek 모델**: DeepSeek-LLM 7B 모델 활용
- **스트리밍 응답**: 실시간 타이핑 효과로 응답 표시
- **한국어 지원**: 한국어 응답 강제 설정

**주요 기능**:
- **Ollama 연동**: `langchain-ollama`를 통한 로컬 LLM 연결
- **DeepSeek 모델**: `deepseek-llm:7b-chat` 모델 사용
- **스트리밍 응답**: `llm.stream()`을 통한 실시간 응답 처리
- **한국어 강제**: 시스템 메시지를 통한 한국어 응답 보장

**주요 특징**:
- **로컬 실행**: 인터넷 연결 없이도 AI 채팅 가능
- **개인정보 보호**: 데이터가 로컬에서만 처리됨
- **비용 절약**: API 호출 비용 없이 무제한 사용 가능
- **커스터마이징**: 다양한 오픈소스 모델 선택 가능

**실행 방법**:
```bash
# Ollama 설치 및 DeepSeek 모델 다운로드
ollama pull deepseek-llm:7b-chat

# Streamlit 앱 실행
streamlit run chap11/deepseek_simple_chatbot.py
```

**사용 예시**:
```
사용자: 안녕하세요, 오늘 날씨가 어때요?
AI: 안녕하세요! 오늘 날씨에 대해 말씀드리기에는 제가 실시간 날씨 정보에 접근할 수 없어서 정확한 답변을 드릴 수 없습니다. 

하지만 일반적으로 날씨 정보를 확인하는 방법을 알려드릴 수 있어요:
1. 날씨 앱이나 웹사이트 확인
2. 기상청 공식 사이트 방문
3. 지역별 날씨 검색

혹시 특정 지역의 날씨가 궁금하시다면, 해당 지역을 말씀해 주시면 더 구체적인 정보를 찾아드릴 수 있을 것 같습니다.
```

### Chapter 12: LangGraph 기반 AI Agent

#### 12.1 LangGraph 메모리 시스템 (`langgraph_memory.py`)

- **상태 관리**: `StateGraph`를 활용한 대화 상태 관리
- **메모리 지속성**: `MemorySaver`를 통한 대화 기록 저장
- **그래프 기반 처리**: 노드와 엣지로 구성된 AI 처리 파이프라인
- **체크포인트**: 대화 세션별 독립적인 메모리 관리

**주요 기능**:
- **StateGraph**: `State` 클래스를 통한 대화 상태 정의
- **메모리 저장**: `MemorySaver`를 활용한 대화 기록 지속성
- **노드 기반 처리**: `generate` 노드를 통한 AI 응답 생성
- **세션 관리**: `thread_id`를 통한 독립적인 대화 세션

**주요 특징**:
- **상태 기반 아키텍처**: TypedDict를 활용한 타입 안전한 상태 관리
- **메모리 지속성**: 대화 세션 간 메시지 히스토리 유지
- **그래프 구조**: 확장 가능한 노드 기반 처리 시스템
- **체크포인트 시스템**: 대화 상태의 안전한 저장 및 복원

**실행 방법**:
```bash
python chap12/langgraph_memory.py
```

**사용 예시**:
```python
# 상태 정의
class State(TypedDict):
    messages: Annotated[list[str], add_messages]

# 그래프 구성
graph_builder = StateGraph(State)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)

# 메모리와 함께 컴파일
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# 대화 실행
for event in graph.stream({"messages": [HumanMessage(user_input)]}, config):
    event["messages"][-1].pretty_print()
```

#### 12.2 LangGraph 기본 채팅 (`langgraph_simple_chatbot.ipynb`)

- **Jupyter 노트북**: 인터랙티브한 LangGraph 학습 환경
- **기본 채팅 시스템**: LangGraph의 핵심 개념 이해
- **상태 관리**: 대화 상태의 구조화 및 관리 방법

**주요 기능**:
- **LangGraph 기본 구조**: 그래프 기반 AI 처리 시스템 이해
- **상태 관리**: 대화 상태의 정의 및 업데이트 방법
- **노드 구성**: AI 처리 파이프라인의 노드 구성 및 연결

**실행 방법**:
```bash
jupyter notebook chap12/langgraph_simple_chatbot.ipynb
```

#### 12.3 LangGraph 도구 연동 시스템 (`langgraph_tools.ipynb`)

- **도구 통합**: LangGraph와 LangChain 도구의 결합
- **고급 AI Agent**: 도구를 활용한 능동적인 AI 에이전트
- **복잡한 워크플로우**: 다단계 AI 처리 파이프라인 구현

**주요 기능**:
- **도구 바인딩**: LangGraph에서 LangChain 도구 활용
- **워크플로우 관리**: 복잡한 AI 처리 과정의 체계적 관리
- **상태 기반 도구 실행**: 대화 상태에 따른 동적 도구 선택

**실행 방법**:
```bash
jupyter notebook chap12/langgraph_tools.ipynb
```

### Chapter 13: LangGraph + RAG 통합 시스템

#### 13.1 LangGraph RAG 통합 시스템 (`rag_with_langgraph.ipynb`)

- **고급 RAG**: LangGraph와 RAG의 결합으로 더욱 지능적인 AI Agent
- **FAISS 벡터 저장소**: Chroma 대신 FAISS를 활용한 고성능 벡터 검색
- **그래프 기반 RAG**: LangGraph의 상태 관리와 RAG의 검색 기능 결합
- **대화 맥락 이해**: 복잡한 대화 흐름에서도 정확한 문서 검색 및 응답

**주요 기능**:
- **FAISS 벡터 저장소**: 고성능 벡터 검색을 위한 FAISS 활용
- **문서 처리 파이프라인**: PDF 로딩부터 텍스트 분할까지 자동화
- **그래프 기반 상태 관리**: LangGraph를 통한 RAG 상태 관리
- **고급 검색 알고리즘**: 복잡한 쿼리에 대한 정확한 문서 검색

**주요 특징**:
- **성능 최적화**: FAISS를 통한 빠른 벡터 검색
- **확장 가능한 아키텍처**: 대용량 문서 처리 지원
- **상태 기반 RAG**: 대화 맥락을 고려한 문서 검색
- **그래프 기반 처리**: 복잡한 AI 워크플로우 지원

**학습 데이터**:
- **뉴욕시 전략 계획서**: OneNYC 2050 Strategic Plan 요약본
- **서울시 2040 계획서**: 2040 Seoul Plan

**실행 방법**:
```bash
jupyter notebook chap13/rag_with_langgraph.ipynb
```

**사용 예시**:
```python
# PDF 문서 처리
def read_pdf_and_split_text(pdf_path, chunk_size=1000, chunk_overlap=100):
    pdf_loader = PyPDFLoader(pdf_path)
    data_from_pdf = pdf_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(data_from_pdf)
    return splits

# FAISS 벡터 저장소 생성
embedding = OpenAIEmbeddings(model='text-embedding-3-large')
vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding)

# LangGraph와 RAG 결합
# (구체적인 구현은 노트북에서 확인 가능)
```

### Chapter 14: 다중 AI Agent 협업 시스템

#### 14.1 다중 AI Agent를 활용한 책 작성 시스템 (`book_writer.py`)

- **다중 Agent 협업**: Supervisor, Content Strategist, Communicator의 협업을 통한 책 작성
- **LangGraph 기반 워크플로우**: 상태 기반 그래프 구조로 복잡한 작업 흐름 관리
- **자동화된 책 작성**: 사용자 요구사항에 따른 자동 목차 생성 및 책 내용 구성
- **상태 지속성**: 작업 진행 상황과 목차를 파일로 저장하여 지속적인 작업 가능

**주요 기능**:
- **Supervisor**: AI 팀의 작업을 관리하고 지도하는 관리자 역할
- **Content Strategist**: 사용자 요구사항 분석 및 책 목차(outline) 작성
- **Communicator**: AI 팀의 진행 상황을 사용자에게 보고하고 의견 수렴
- **자동 라우팅**: 작업 상황에 따라 적절한 Agent로 작업 전달
- **목차 관리**: Markdown 형식의 목차 자동 생성 및 저장

**주요 특징**:
- **역할 기반 분업**: 각 Agent가 명확한 역할과 책임을 가짐
- **상태 기반 워크플로우**: LangGraph의 StateGraph를 활용한 체계적인 상태 관리
- **스트리밍 응답**: 실시간으로 AI 응답을 생성하여 사용자 경험 향상
- **자동화된 워크플로우**: 사용자 입력에 따른 자동 작업 분배 및 실행
- **시각화**: Mermaid PNG 형식으로 LangGraph 구조 시각화

**실행 방법**:
```bash
python chap14/book_writer.py
```

**사용 예시**:
```
사용자: HYBE와 JYP를 비교하는 책을 써줘

[Supervisor] 사용자의 요구사항이 명확해졌습니다. Content Strategist를 통해 책의 목차를 작성하겠습니다.

[Content Strategist] 사용자의 요구사항을 분석하여 HYBE와 JYP 비교 책의 목차를 작성하겠습니다.

1. 서문
2. 엔터테인먼트 산업의 배경
3. HYBE와 JYP 소개
4. 비전과 철학
5. 아티스트와 콘텐츠 전략
...

[Communicator] 목차 작성이 완료되었습니다. 사용자님의 의견을 들려주세요.
```

#### 14.2 Task 모델 및 유틸리티 시스템

**Task 모델 (`models.py`)**:
- **Pydantic 기반**: 타입 안전성을 위한 Task 데이터 모델 정의
- **Agent 분류**: content_strategist, communicator 등 작업 수행 Agent 구분
- **작업 상태 추적**: 작업 완료 여부, 완료 시간 등 상세한 작업 이력 관리

**유틸리티 시스템 (`utils.py`)**:
- **상태 저장**: AI Agent의 작업 상태를 JSON 형식으로 저장
- **목차 관리**: Markdown 형식의 책 목차 저장 및 로드
- **파일 시스템**: 데이터 디렉토리 자동 생성 및 파일 관리

**주요 특징**:
- **모듈화된 구조**: 각 기능을 독립적인 모듈로 분리하여 재사용성 향상
- **데이터 지속성**: 작업 진행 상황과 결과를 파일로 저장하여 중단된 작업 재개 가능
- **유연한 확장**: 새로운 Agent 타입과 작업 유형을 쉽게 추가 가능

## 🔧 주요 기능

### 1. 환경 변수 관리
- `python-dotenv`를 사용한 안전한 API 키 관리
- `.env` 파일을 통한 환경 설정

### 2. 대화 시스템
- **싱글턴**: 각 대화가 독립적
- **멀티턴**: 대화 히스토리 유지
- **웹 인터페이스**: 브라우저 기반 채팅

### 3. GPT Functions & LangChain Tools
- **함수 정의**: GPT가 호출할 수 있는 함수 스키마 정의
- **함수 호출**: AI가 필요에 따라 함수를 자동 호출
- **결과 처리**: 함수 실행 결과를 AI 응답에 통합
- **LangChain Tools**: `@tool` 데코레이터를 사용한 체계적인 도구 관리
- **Pydantic 모델**: 타입 안전성을 위한 입력값 검증
- **메시지 히스토리**: `RunnableWithMessageHistory`를 통한 대화 기록 관리
- **세션 관리**: 사용자별 독립적인 대화 세션 지원
- **재귀적 스트리밍**: 도구 호출 후 최종 응답도 스트리밍으로 처리

### 4. RAG (Retrieval-Augmented Generation)
- **문서 처리**: PDF 문서 로딩 및 텍스트 분할
- **벡터 데이터베이스**: Chroma를 활용한 임베딩 저장 및 검색
- **유사도 검색**: 쿼리와 관련된 문서 청크 검색
- **문맥 기반 응답**: 검색된 문서를 바탕으로 정확한 답변 생성
- **실시간 검색**: 사용자 질문에 관련된 문서를 실시간으로 검색

### 5. 웹 검색 및 유튜브 연동
- **DuckDuckGo 검색**: 프라이버시 중심의 웹 검색 API 활용
- **유튜브 검색**: 동영상 검색 및 자막 추출을 통한 내용 분석
- **통합 도구 시스템**: 웹 검색, 유튜브, 시간 조회를 하나의 인터페이스에서 활용
- **검색 기간 설정**: 과거 1주일, 1개월, 1년 등 검색 기간 지정 가능

### 6. 로컬 LLM (DeepSeek)
- **Ollama 연동**: 로컬에서 실행되는 언어 모델 시스템
- **DeepSeek 모델**: DeepSeek-LLM 7B 모델을 활용한 한국어 채팅
- **개인정보 보호**: 데이터가 로컬에서만 처리되어 프라이버시 보장
- **비용 절약**: API 호출 비용 없이 무제한 AI 채팅 가능

### 7. LangGraph 기반 AI Agent
- **상태 관리**: `StateGraph`를 활용한 체계적인 대화 상태 관리
- **메모리 지속성**: `MemorySaver`를 통한 대화 기록의 안전한 저장 및 복원
- **그래프 기반 처리**: 노드와 엣지로 구성된 확장 가능한 AI 처리 파이프라인
- **체크포인트 시스템**: 대화 세션별 독립적인 메모리 관리
- **도구 통합**: LangGraph와 LangChain 도구의 결합으로 더욱 능동적인 AI Agent

### 8. LangGraph + RAG 통합 시스템
- **고급 RAG**: LangGraph의 상태 관리와 RAG의 검색 기능을 결합한 지능형 시스템
- **FAISS 벡터 저장소**: Chroma 대신 FAISS를 활용한 고성능 벡터 검색
- **그래프 기반 상태 관리**: 복잡한 대화 흐름에서도 정확한 문서 검색 및 응답
- **확장 가능한 아키텍처**: 대용량 문서 처리 및 복잡한 AI 워크플로우 지원

### 9. 다중 AI Agent 협업 시스템
- **역할 기반 분업**: Supervisor, Content Strategist, Communicator의 명확한 역할 분담
- **LangGraph 워크플로우**: 상태 기반 그래프 구조로 복잡한 작업 흐름 자동화
- **자동화된 책 작성**: 사용자 요구사항에 따른 자동 목차 생성 및 책 내용 구성
- **상태 지속성**: 작업 진행 상황과 결과를 파일로 저장하여 중단된 작업 재개 가능
- **시각화**: Mermaid PNG 형식으로 LangGraph 구조 시각화

### 9. 모델 설정
- GPT-4o 모델 사용
- Temperature 조절을 통한 응답 창의성 제어
- 시스템 메시지를 통한 AI 페르소나 설정

## 📝 사용 예시

### 기본 대화 예시
```
사용자: 안녕 내 이름은 이시현이야
AI: 안녕하세요, 시현님! 만나서 반갑습니다. 오늘 어떻게 도와드릴까요?

사용자: 내 이름이 뭘까?
AI: 시현님이라고 소개해 주셨죠! 제가 대화 상태를 기억하고 있어서 이전 대화 내용을 모두 알고 있습니다.

사용자: 그럼 제가 지금까지 몇 번 대화했나요?
AI: 현재 대화 세션에서 총 3번의 메시지를 주고받았습니다. 

- 첫 번째: 김철수님이 이름을 소개해 주셨습니다
- 두 번째: 제가 인사를 했습니다  
- 세 번째: 김철수님이 제가 이름을 기억하는지 확인하셨습니다

LangGraph의 상태 관리 시스템 덕분에 모든 대화 내용을 체계적으로 추적하고 있습니다.
```

### 함수 호출 예시

#### 시간 조회
```
사용자: 서울 시간이 몇 시야?
AI: 서울의 현재 시간은 2025년 8월 31일 오후 3시 45분입니다.

사용자: 뉴욕과 도쿄 시간도 알려줘
AI: 현재 시각은 다음과 같습니다:
- 뉴욕: 2025년 8월 31일 오전 2시 45분
- 도쿄: 2025년 8월 31일 오후 4시 45분
```

#### 주식 정보 조회
```
사용자: 애플 주식 정보 알려줘
AI: 애플(AAPL)의 주가 정보를 조회해드리겠습니다.

사용자: 마이크로소프트 주가와 현재 뉴욕 시간 알려줘
AI: 마이크로소프트(MSFT)의 최근 주가는 다음과 같습니다:
- 오픈가: $508.66
- 고가: $509.6
- 저가: $504.49
- 종가: $506.69
- 거래량: 20,954,200

현재 미국 뉴욕의 시간은 2025년 9월 1일 04:41:41입니다.
```

### RAG 시스템 예시

#### 문서 기반 질의응답
```
사용자: 뉴욕시의 기후 변화 대응 정책은 무엇인가요?
AI: 뉴욕시의 기후 변화 대응 정책에 대해 OneNYC 2050 전략 계획서를 참조하여 답변드리겠습니다.

뉴욕시는 다음과 같은 주요 기후 변화 대응 정책을 추진하고 있습니다:

1. **탄소 중립 목표**: 2050년까지 탄소 중립 달성
2. **재생에너지 확대**: 태양광, 풍력 등 청정에너지 확대
3. **건물 에너지 효율화**: 기존 건물의 에너지 효율 개선
4. **대중교통 확대**: 대중교통 이용 촉진 및 전기차 보급
5. **해안 보호**: 기후 변화에 따른 해수면 상승 대비

이러한 정책들은 OneNYC 2050 전략의 핵심 구성 요소로, 지속 가능한 도시 발전을 위한 종합적인 접근 방식을 제시합니다.

#### Streamlit RAG 채팅 시스템
```
사용자: 서울시의 2040 계획에서 교통 정책은 어떻게 되어있나요?

[문서 검색 중...]
- 2040_seoul_plan.pdf (페이지 45-52)
- 2040_seoul_plan.pdf (페이지 78-85)

AI: 서울시 2040 계획서를 참조하여 교통 정책에 대해 답변드리겠습니다.

서울시 2040 계획의 주요 교통 정책은 다음과 같습니다:

1. **대중교통 중심 도시**: 지하철, 버스 등 대중교통 확충
2. **친환경 교통**: 전기차, 수소차 등 친환경 차량 보급 확대
3. **보행자 중심**: 보행자 우선 도시 조성
4. **스마트 교통**: AI 기반 교통 관리 시스템 구축

[참조 문서: 2040_seoul_plan.pdf, 페이지: 45-52, 78-85]
```

### 웹 검색 및 유튜브 연동 예시

#### 웹 검색
```
사용자: 최신 AI 기술 동향에 대해 검색해줘
AI: 웹 검색을 통해 최신 AI 기술 동향에 대한 정보를 찾아보겠습니다.

검색 결과:
1. **생성형 AI의 발전**: GPT-4, Claude, Gemini 등 대형 언어 모델의 성능 향상
2. **멀티모달 AI**: 텍스트, 이미지, 음성, 비디오를 통합 처리하는 AI 시스템
3. **AI 에이전트**: 자율적으로 작업을 수행하는 AI 에이전트 기술
4. **AI 규제 및 윤리**: AI 발전에 따른 규제 프레임워크 및 윤리적 고려사항

[검색 기간: 최근 1개월, 지역: 한국]
```

#### 유튜브 검색 및 자막 분석
```
사용자: AI 에이전트에 대한 유튜브 동영상을 찾아서 요약해줘
AI: AI 에이전트에 대한 유튜브 동영상을 검색하고 자막을 분석하여 요약해드리겠습니다.

검색된 동영상:
1. **"AI 에이전트의 미래"** (5분 23초)
   - 내용: AI 에이전트가 일상생활에서 수행할 수 있는 작업들
   - 핵심: 자율성, 학습 능력, 인간과의 협업

2. **"LangChain으로 AI 에이전트 만들기"** (4분 15초)
   - 내용: LangChain 프레임워크를 활용한 AI 에이전트 개발 방법
   - 핵심: 도구 연동, 메모리 관리, 워크플로우 구성

[자막 분석 완료: 총 2개 동영상, 한국어 자막 우선 처리]
```

### 로컬 LLM (DeepSeek) 예시

#### 개인정보 보호 채팅
```
사용자: 내 개인정보를 수집하지 말고 대화해줘
AI: 네, 걱정하지 마세요! 저는 Ollama를 통해 로컬에서 실행되는 AI 모델이에요. 

제가 하는 일은:
- 모든 대화가 여러분의 컴퓨터에서만 처리됩니다
- 인터넷으로 데이터를 전송하지 않습니다
- 대화 내용이 외부 서버에 저장되지 않습니다
- 완전히 프라이버시가 보장됩니다

그래서 마음 편하게 대화하실 수 있어요. 어떤 주제로 이야기하고 싶으신가요?
```

### LangGraph 기반 AI Agent 예시

#### 상태 기반 대화 관리
```
사용자: 안녕하세요, 제 이름은 김철수입니다
AI: 안녕하세요, 김철수님! 만나서 반갑습니다. 오늘 어떻게 도와드릴까요?

사용자: 내 이름이 뭔지 기억하시나요?
AI: 네, 김철수님이라고 소개해 주셨죠! 제가 대화 상태를 기억하고 있어서 이전 대화 내용을 모두 알고 있습니다.

사용자: 그럼 제가 지금까지 몇 번 대화했나요?
AI: 현재 대화 세션에서 총 3번의 메시지를 주고받았습니다. 

- 첫 번째: 김철수님이 이름을 소개해 주셨습니다
- 두 번째: 제가 인사를 했습니다  
- 세 번째: 김철수님이 제가 이름을 기억하는지 확인하셨습니다

LangGraph의 상태 관리 시스템 덕분에 모든 대화 내용을 체계적으로 추적하고 있습니다.
```

### LangGraph + RAG 통합 시스템 예시

#### 복잡한 대화 맥락에서의 문서 검색
```
사용자: 서울시 2040 계획에서 교통 정책은 어떻게 되어있나요?
AI: 서울시 2040 계획서를 참조하여 교통 정책에 대해 답변드리겠습니다.

[문서 검색 중... LangGraph 상태 업데이트]
- 검색된 문서: 2040_seoul_plan.pdf (페이지 45-52, 78-85)
- 관련성 점수: 0.89, 0.76, 0.72

서울시 2040 계획의 주요 교통 정책은 다음과 같습니다:

1. **대중교통 중심 도시**: 지하철, 버스 등 대중교통 확충
2. **친환경 교통**: 전기차, 수소차 등 친환경 차량 보급 확대
3. **보행자 중심**: 보행자 우선 도시 조성
4. **스마트 교통**: AI 기반 교통 관리 시스템 구축

[LangGraph 상태: 교통 정책 관련 대화 컨텍스트 저장됨]

사용자: 그럼 환경 정책은 어떻게 되어있나요?
AI: 서울시 2040 계획서에서 환경 정책에 대해 추가로 검색해보겠습니다.

[이전 교통 정책 컨텍스트를 고려한 검색...]
- 검색된 문서: 2040_seoul_plan.pdf (페이지 23-30, 67-74)
- 관련성 점수: 0.91, 0.85

환경 정책은 교통 정책과 연계되어 다음과 같이 구성되어 있습니다:

1. **탄소 중립 목표**: 2040년까지 탄소 중립 달성
2. **청정에너지 확대**: 태양광, 풍력 등 재생에너지 확대
3. **친환경 교통**: 전기차, 수소차 보급 확대 (교통 정책과 연계)
4. **녹지 확대**: 도시 녹지율 30% 달성

[LangGraph 상태: 교통-환경 정책 통합 컨텍스트 업데이트됨]
```

### 다중 AI Agent 협업 시스템 예시

#### 책 작성 워크플로우
```
사용자: HYBE와 JYP를 비교하는 책을 써줘

[Supervisor] 사용자의 요구사항이 명확해졌습니다. Content Strategist를 통해 책의 목차를 작성하겠습니다.

[Content Strategist] 사용자의 요구사항을 분석하여 HYBE와 JYP 비교 책의 목차를 작성하겠습니다.

1. 서문
2. 엔터테인먼트 산업의 배경
3. HYBE와 JYP 소개
4. 비전과 철학
5. 아티스트와 콘텐츠 전략
6. 글로벌 진출과 전략
7. 팬덤 문화
8. 기술의 활용
9. 사회적 책임
10. 재무 성과 및 비즈니스 모델
11. 향후 전망과 도전 과제
12. 결론
13. 부록

[Communicator] 목차 작성이 완료되었습니다. 사용자님의 의견을 들려주세요.

사용자: 목차가 좋네요. 이제 각 장의 내용을 작성해주세요.

[Supervisor] 사용자가 목차에 만족하고 있습니다. Content Strategist를 통해 각 장의 상세 내용을 작성하겠습니다.
```

### 종료 방법
- 콘솔 애플리케이션: `exit` 입력
- Streamlit: 브라우저 창 닫기

## 🛠️ 기술 스택

- **Python 3.x**
- **OpenAI API**: GPT-4o 모델
- **Streamlit**: 웹 인터페이스
- **python-dotenv**: 환경 변수 관리
- **pytz**: 시간대 처리
- **yfinance**: Yahoo Finance API (주식 데이터)
- **LangChain**: AI 애플리케이션 개발 프레임워크
- **Pydantic**: 데이터 검증 및 설정 관리
- **Jupyter Notebook**: 인터랙티브 개발 환경
- **Chroma**: 벡터 데이터베이스
- **langchain-community**: 문서 로더 및 텍스트 분할기
- **langchain-chroma**: Chroma 벡터 저장소 연동
- **langchain-ollama**: Ollama를 통한 로컬 LLM 연동
- **duckduckgo-search**: DuckDuckGo 웹 검색 API
- **youtube-search**: 유튜브 동영상 검색
- **youtube-transcript-api**: 유튜브 자막 추출
- **faiss-cpu**: FAISS 벡터 데이터베이스 (CPU 버전)
- **langgraph**: LangGraph 기반 AI Agent 개발 프레임워크
- **mermaid**: LangGraph 구조 시각화

## 📖 학습 목표

1. **AI Agent 기본 개념 이해**
   - 대화형 AI의 동작 원리
   - 메모리와 맥락의 중요성

2. **API 활용 능력 향상**
   - OpenAI API 사용법
   - 응답 처리 및 에러 핸들링

3. **GPT Functions & LangChain Tools 활용**
   - 함수 호출 메커니즘 이해
   - 외부 함수와 AI의 연동 방법
   - 스트리밍 응답과 함수 호출의 조합
   - LangChain 프레임워크를 통한 체계적인 도구 관리
   - Pydantic을 활용한 타입 안전성 확보
   - 메시지 히스토리 관리 및 세션 기반 대화
   - 재귀적 스트리밍과 도구 호출의 조합

4. **RAG (Retrieval-Augmented Generation) 시스템 구현**
   - 문서 처리 및 텍스트 분할 기법
   - 벡터 데이터베이스 구축 및 관리
   - 유사도 검색 알고리즘 이해
   - 문맥 기반 AI 응답 생성
   - Chroma 벡터 저장소 활용

5. **웹 검색 및 유튜브 연동 시스템 구현**
   - DuckDuckGo API를 활용한 실시간 웹 검색
   - YouTube Transcript API를 통한 동영상 자막 추출 및 분석
   - 통합 도구 시스템을 통한 다중 정보 소스 활용
   - 검색 기간 설정 및 지역별 검색 결과 최적화

6. **로컬 LLM 시스템 구현**
   - Ollama를 통한 로컬 언어 모델 실행
   - DeepSeek-LLM 모델을 활용한 한국어 AI 채팅
   - 개인정보 보호 및 비용 절약을 위한 로컬 AI 시스템
   - 다양한 오픈소스 모델 선택 및 커스터마이징

7. **LangGraph 기반 AI Agent 시스템**
   - LangGraph를 활용한 상태 기반 AI 처리 시스템
   - 그래프 구조를 통한 복잡한 AI 워크플로우 구현
   - 메모리 지속성 및 체크포인트 시스템을 통한 안정적인 대화 관리
   - 노드 기반 확장 가능한 AI 처리 파이프라인 구축

8. **LangGraph + RAG 통합 시스템**
   - LangGraph와 RAG의 결합으로 더욱 지능적인 AI Agent 구현
   - FAISS 벡터 저장소를 활용한 고성능 문서 검색
   - 복잡한 대화 맥락에서의 정확한 문서 검색 및 응답 생성
   - 확장 가능한 아키텍처를 통한 대용량 문서 처리 시스템

9. **다중 AI Agent 협업 시스템**
   - 다중 AI Agent의 역할 기반 분업 및 협업 시스템 구현
   - LangGraph를 활용한 복잡한 워크플로우 자동화
   - 자동화된 책 작성 및 콘텐츠 생성 시스템
   - 상태 지속성과 시각화를 통한 작업 진행 상황 관리

10. **실용적 애플리케이션 개발**
   - 콘솔 기반 채팅봇
   - 웹 기반 인터페이스
   - 함수 호출 기능이 포함된 AI 시스템
   - 실시간 데이터 조회 시스템 (주식, 시간 등)
   - Jupyter 노트북을 활용한 인터랙티브 AI 개발
   - LangChain을 활용한 고급 AI 애플리케이션
   - 세션 기반 대화 시스템 및 메시지 히스토리 관리
   - RAG 시스템을 활용한 문서 기반 AI 챗봇
   - 벡터 데이터베이스를 활용한 지식 검색 시스템
   - Streamlit 기반 RAG 채팅 애플리케이션
   - 모듈화된 RAG 시스템 컴포넌트 개발
   - 웹 검색 및 유튜브 연동을 통한 실시간 정보 수집 시스템
   - 로컬 LLM을 활용한 개인정보 보호 AI 채팅 시스템
   - LangGraph 기반 상태 관리 및 메모리 지속성 시스템
- FAISS 벡터 저장소를 활용한 고성능 RAG 시스템
- 다중 AI Agent 협업을 통한 자동화된 콘텐츠 생성 시스템
- 역할 기반 분업과 워크플로우 자동화를 통한 복잡한 작업 처리

## 🚀 고급 기능

### 스트리밍 응답 처리
- **실시간 타이핑 효과**: 응답이 실시간으로 타이핑되듯이 표시
- **함수 호출 후 스트리밍**: 툴 실행 후 최종 응답도 스트리밍으로 처리
- **사용자 경험 향상**: 더 자연스러운 대화 인터페이스

### 다중 함수 호출
- **동시 실행**: 여러 함수를 동시에 호출하여 효율적인 데이터 수집
- **메시지 순서 최적화**: OpenAI API 요구사항에 맞는 메시지 구조
- **안정적인 처리**: 함수 호출 오류 방지 및 안정적인 응답 생성

### 실시간 데이터 연동
- **Yahoo Finance API**: 실시간 주식 데이터 조회
- **다양한 시간대**: 전 세계 시간대 지원
- **마크다운 형식**: 표 형태의 데이터를 깔끔하게 표시

### LangChain 고급 기능
- **메시지 히스토리 관리**: `RunnableWithMessageHistory`를 통한 대화 기록 유지
- **세션 기반 대화**: 사용자별 독립적인 대화 세션 지원
- **도구 바인딩**: `llm.bind_tools()`를 통한 모델과 도구의 연결
- **재귀적 스트리밍**: 도구 호출 후 최종 응답도 스트리밍으로 처리
- **메시지 타입 구분**: 시스템, 사용자, AI, 도구 메시지를 구분하여 표시

### 웹 검색 및 유튜브 연동
- **DuckDuckGo 검색**: 프라이버시 중심의 웹 검색 API 활용
- **유튜브 검색**: 동영상 검색 및 자막 추출을 통한 내용 분석
- **통합 도구 시스템**: 웹 검색, 유튜브, 시간 조회를 하나의 인터페이스에서 활용
- **검색 기간 설정**: 과거 1주일, 1개월, 1년 등 검색 기간 지정 가능

### 로컬 LLM (DeepSeek)
- **Ollama 연동**: 로컬에서 실행되는 언어 모델 시스템
- **DeepSeek 모델**: DeepSeek-LLM 7B 모델을 활용한 한국어 채팅
- **개인정보 보호**: 데이터가 로컬에서만 처리되어 프라이버시 보장
- **비용 절약**: API 호출 비용 없이 무제한 AI 채팅 가능

### LangGraph 기반 AI Agent
- **상태 관리**: `StateGraph`를 활용한 체계적인 대화 상태 관리
- **메모리 지속성**: `MemorySaver`를 통한 대화 기록의 안전한 저장 및 복원
- **그래프 기반 처리**: 노드와 엣지로 구성된 확장 가능한 AI 처리 파이프라인
- **체크포인트 시스템**: 대화 세션별 독립적인 메모리 관리
- **도구 통합**: LangGraph와 LangChain 도구의 결합으로 더욱 능동적인 AI Agent

### LangGraph + RAG 통합 시스템
- **고급 RAG**: LangGraph의 상태 관리와 RAG의 검색 기능을 결합한 지능형 시스템
- **FAISS 벡터 저장소**: Chroma 대신 FAISS를 활용한 고성능 벡터 검색
- **그래프 기반 상태 관리**: 복잡한 대화 흐름에서도 정확한 문서 검색 및 응답
- **확장 가능한 아키텍처**: 대용량 문서 처리 및 복잡한 AI 워크플로우 지원

### 다중 AI Agent 협업 시스템
- **역할 기반 분업**: Supervisor, Content Strategist, Communicator의 명확한 역할 분담
- **LangGraph 워크플로우**: 상태 기반 그래프 구조로 복잡한 작업 흐름 자동화
- **자동화된 콘텐츠 생성**: 사용자 요구사항에 따른 자동 목차 생성 및 책 내용 구성
- **상태 지속성**: 작업 진행 상황과 결과를 파일로 저장하여 중단된 작업 재개 가능
- **시각화**: Mermaid PNG 형식으로 LangGraph 구조 시각화

### RAG 시스템
- **문서 처리 파이프라인**: PDF 로딩부터 텍스트 분할까지 자동화
- **벡터 임베딩**: OpenAI 임베딩 모델을 사용한 고품질 벡터화
- **유사도 검색**: 코사인 유사도를 통한 정확한 문서 검색
- **문맥 기반 생성**: 검색된 문서를 바탕으로 사실에 기반한 답변
- **확장 가능한 아키텍처**: 다양한 문서 형식과 벡터 저장소 지원

## 🔒 보안 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요
- API 키는 안전하게 관리하고 공개하지 마세요
- `.gitignore`에 `.env`가 포함되어 있는지 확인하세요

## 🤝 기여하기

1. 이 저장소를 포크하세요
2. 새로운 기능 브랜치를 생성하세요 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/AmazingFeature`)
5. Pull Request를 생성하세요

## 📄 라이선스

이 프로젝트는 학습 목적으로 제작되었습니다.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.

---

**Happy Learning! 🚀**
