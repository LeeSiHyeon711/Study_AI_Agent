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
   pip install openai python-dotenv streamlit
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

## 🔧 주요 기능

### 1. 환경 변수 관리
- `python-dotenv`를 사용한 안전한 API 키 관리
- `.env` 파일을 통한 환경 설정

### 2. 대화 시스템
- **싱글턴**: 각 대화가 독립적
- **멀티턴**: 대화 히스토리 유지
- **웹 인터페이스**: 브라우저 기반 채팅

### 3. 모델 설정
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

### 종료 방법
- 콘솔 애플리케이션: `exit` 입력
- Streamlit: 브라우저 창 닫기

## 🛠️ 기술 스택

- **Python 3.x**
- **OpenAI API**: GPT-4o 모델
- **Streamlit**: 웹 인터페이스
- **python-dotenv**: 환경 변수 관리

## 📖 학습 목표

1. **AI Agent 기본 개념 이해**
   - 대화형 AI의 동작 원리
   - 메모리와 맥락의 중요성

2. **API 활용 능력 향상**
   - OpenAI API 사용법
   - 응답 처리 및 에러 핸들링

3. **실용적 애플리케이션 개발**
   - 콘솔 기반 채팅봇
   - 웹 기반 인터페이스

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
