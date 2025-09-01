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
│   └── langchain_tool.ipynb # LangChain을 활용한 도구 연동 예제
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
   pip install openai python-dotenv streamlit pytz yfinance langchain-openai langchain-core
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

**파일**: `chap08/langchain_tool.ipynb`

- **LangChain 프레임워크**: OpenAI API를 더 체계적으로 활용하는 프레임워크
- **Tool 시스템**: LangChain의 `@tool` 데코레이터를 사용한 함수 정의
- **Pydantic 모델**: 타입 안전성을 위한 입력값 검증 및 구조화
- **Jupyter 노트북**: 인터랙티브한 학습 환경

**주요 기능**:
- **시간 조회 도구**: `get_current_time` 함수로 다양한 타임존의 현재 시간 조회
- **주식 데이터 조회**: `get_yf_stock_history` 함수로 Yahoo Finance API 연동
- **도구 바인딩**: `llm.bind_tools()`를 통한 모델과 도구의 연결
- **메시지 처리**: LangChain의 메시지 시스템을 통한 대화 관리

**학습 내용**:
- LangChain의 기본 구조와 개념
- `@tool` 데코레이터를 사용한 함수 정의
- Pydantic을 활용한 입력값 검증
- 도구 호출 및 결과 처리 과정
- Jupyter 노트북에서의 인터랙티브 개발

**실행 방법**:
```bash
jupyter notebook chap08/langchain_tool.ipynb
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

### 4. 모델 설정
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

4. **실용적 애플리케이션 개발**
   - 콘솔 기반 채팅봇
   - 웹 기반 인터페이스
   - 함수 호출 기능이 포함된 AI 시스템
   - 실시간 데이터 조회 시스템 (주식, 시간 등)
   - Jupyter 노트북을 활용한 인터랙티브 AI 개발

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
