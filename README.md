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
│       ├── OneNYC_2050_Strategic_Plan.pdf # 뉴욕시 전략 계획서
│       └── 2040_seoul_plan.pdf # 서울시 2040 계획서
├── chroma_store/          # Chroma 벡터 데이터베이스 저장소
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

### 5. 모델 설정
- GPT-4o 모델 사용
- Temperature 조절을 통한 응답 창의성 제어
- 시스템 메시지를 통한 AI 페르소나 설정

## 📝 사용 예시

### 기본 대화 예시
```
사용자: 안녕 내 이름은 이시현이야
AI: 안녕하세요, 시현님! 만나서 반갑습니다. 오늘 어떻게 도와드릴까요?

사용자: 내 이름이 뭘까?
AI: 시현님이라고 소개해 주셨는데, 더 구체적으로 말씀해주시면...
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

5. **실용적 애플리케이션 개발**
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
