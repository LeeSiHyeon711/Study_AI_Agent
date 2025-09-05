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
├── chap14/                 # 14장: 고급 다중 AI Agent 협업 시스템
│   ├── book_writer.py     # 다중 AI Agent를 활용한 책 작성 시스템
│   ├── models.py          # Task 모델 정의
│   ├── tools.py           # 웹 검색 및 벡터 검색 도구
│   ├── utils.py           # 상태 저장 및 목차 관리 유틸리티
│   ├── book_writer.png    # LangGraph 시각화 이미지
│   ├── book_writer_graph.mmd # Mermaid 그래프 정의
│   ├── templates/         # 목차 템플릿
│   │   └── outline_template.md # 표준 목차 템플릿
│   └── data/              # 작업 상태 및 목차 데이터
│       ├── state.json     # AI Agent 작업 상태 저장
│       ├── outline.md     # 생성된 책 목차
│       ├── chroma_store/  # 웹 검색 결과 벡터 저장소
│       └── resources_*.json # 웹 검색 결과 저장
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

3. **환경 변수 설정**
   - `.env` 파일을 생성하고 API 키를 설정하세요:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

### 2. API 키 발급

- **OpenAI API**: [OpenAI Platform](https://platform.openai.com/account/api-keys)
- **Tavily API**: [Tavily](https://tavily.com/) (웹 검색용)

## 📚 학습 내용 요약

### 🔰 기초 단계 (2-3장)
- **GPT API 기본 사용법**: OpenAI API 연결 및 기본 대화
- **대화형 AI**: 싱글턴/멀티턴 대화, Streamlit 웹 인터페이스

### 🛠️ 도구 연동 (7-8장)
- **GPT Functions**: 함수 호출을 통한 실시간 데이터 조회
- **LangChain Tools**: 체계적인 도구 관리 및 스트리밍 응답

### 🔍 지식 검색 (9-10장)
- **RAG 시스템**: PDF 문서 기반 질의응답
- **웹 검색**: DuckDuckGo, 유튜브 자막 분석

### 🏠 로컬 실행 (11장)
- **로컬 LLM**: Ollama + DeepSeek 모델로 개인정보 보호

### 🚀 고급 시스템 (12-14장)
- **LangGraph**: 상태 기반 AI Agent 및 메모리 관리
- **RAG 통합**: LangGraph + RAG 결합 시스템
- **다중 Agent**: 7개 Agent 협업을 통한 자동화된 콘텐츠 생성

## 🔧 주요 기능

### 1. **기본 AI 채팅**
- 메모리 기반 대화 시스템
- Streamlit 웹 인터페이스
- 세션별 독립적인 대화 관리

### 2. **실시간 도구 연동**
- 시간 조회 (전 세계 시간대)
- 주식 정보 (Yahoo Finance)
- 웹 검색 (DuckDuckGo, Tavily)
- 유튜브 자막 분석

### 3. **지식 검색 시스템**
- PDF 문서 처리 및 벡터화
- Chroma/FAISS 벡터 데이터베이스
- 문맥 기반 정확한 답변 생성

### 4. **고급 AI Agent**
- LangGraph 기반 상태 관리
- 메모리 지속성 및 체크포인트
- 복잡한 워크플로우 자동화

### 5. **다중 Agent 협업 (7개 Agent)**
- **Business Analyst**: 요구사항 분석 및 작업 결정
- **Supervisor**: 전체 워크플로우 관리
- **Content Strategist**: 목차 및 콘텐츠 전략 수립
- **Outline Reviewer**: 목차 검토 및 개선
- **Vector Search Agent**: 벡터 검색을 통한 정보 수집
- **Web Search Agent**: 웹 검색을 통한 최신 정보 수집
- **Communicator**: 사용자와의 소통 및 피드백 수집

## 🛠️ 기술 스택

- **Python 3.x** + **Streamlit** (웹 인터페이스)
- **OpenAI API** (GPT-4o 모델)
- **LangChain** (AI 애플리케이션 프레임워크)
- **LangGraph** (AI Agent 워크플로우)
- **Chroma/FAISS** (벡터 데이터베이스)
- **Tavily** (고급 웹 검색)
- **Ollama** (로컬 LLM 실행)

## 📖 학습 목표

1. **AI Agent 기본 개념 이해** - 대화형 AI의 동작 원리
2. **API 활용 능력 향상** - OpenAI API 및 다양한 도구 연동
3. **RAG 시스템 구현** - 문서 기반 지식 검색 및 응답 생성
4. **웹 검색 및 멀티미디어 연동** - 실시간 정보 수집 및 분석
5. **로컬 LLM 시스템** - 개인정보 보호 및 비용 절약
6. **LangGraph 기반 AI Agent** - 상태 관리 및 복잡한 워크플로우
7. **다중 Agent 협업** - 7개 Agent의 역할 기반 분업 및 자동화된 작업 처리

## 🎯 핵심 특징

- **단계별 학습**: 기초부터 고급까지 체계적인 학습 경로
- **실용적 예제**: 실제 사용 가능한 AI 애플리케이션 구현
- **다양한 인터페이스**: 콘솔, 웹, 노트북 등 다양한 실행 환경
- **확장 가능한 구조**: 모듈화된 컴포넌트로 새로운 기능 추가 용이
- **상태 지속성**: 작업 진행 상황 저장 및 재개 가능
- **템플릿 시스템**: 표준화된 목차 템플릿으로 일관된 품질 보장

## 🚀 실행 예시

### 기본 채팅
```bash
streamlit run chap03/streamlit_basic.py
```

### RAG 시스템
```bash
streamlit run chap09/rag.py
```

### 고급 다중 Agent 협업
```bash
python chap14/book_writer.py
```

## 🔒 보안 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요
- API 키는 안전하게 관리하고 공개하지 마세요

## 🤝 기여하기

1. 이 저장소를 포크하세요
2. 새로운 기능 브랜치를 생성하세요
3. 변경사항을 커밋하고 Pull Request를 생성하세요

---

**Happy Learning! 🚀**