# AI 사서 보조 싱글 에이전트

**LangChain과 Google Gemini를 활용한 AI Librarian Assistant**

## 1. 프로젝트 소개

본 프로젝트는 LangChain과 Google Gemini를 활용하여 도서 검색 및 도서관 이용 안내를 지원하는 AI 사서 보조 싱글 에이전트이다.

도서관 이용 과정에서 발생할 수 있는 도서 검색의 불편함과 반복적인 이용 문의를 자연어 대화 방식으로 해결하는 것을 목표로 한다.

사용자의 질문 의도를 파악하고, 필요한 도구를 호출하며, 대화 맥락과 사용자 선호도를 기억하여 적절한 응답을 제공하도록 구현하였다.

또한 입력 단계의 Guardrail과 최종 응답 검토 기능을 적용하여 개인정보 침해, 권한 우회 및 부정확한 안내를 방지하고자 하였다.

## 2. 주요 기능

### 2.1 도서 검색
- 도서 제목 및 저자명을 기반으로 도서 검색
- 사전에 등록된 11권의 도서 데이터 활용
- 여러 도서 및 저자 검색 지원
- 검색 결과가 없는 경우 해당 사실을 명확하게 안내

### 2.2 사용자 정보 및 메모리
- 사용자의 도서 취향 및 선호도 저장
- 저장된 사용자 선호도 조회
- SQLite를 활용한 장기 기억 관리
- LangGraph의 InMemorySaver를 활용한 단기 대화 기억 관리

### 2.3 Runtime Context 및 Middleware
- 사서의 이름과 역할을 Runtime Context로 전달
- Middleware를 통해 시스템 정보와 사용자 기억을 모델 입력에 주입
- 사용자 및 서비스 맥락을 반영한 응답 생성

### 2.4 입력 Guardrail
- Gemini를 활용한 의미 기반 사용자 요청 분류
- 도서관 서비스와 무관한 요청 제한
- 타인의 개인정보 및 대출 정보 요청 거부
- 관리자 사칭 및 권한 우회 시도 차단
- 실제 수행하지 않은 작업을 완료했다고 주장하는 요청 제한

### 2.5 최종 응답 검토
- After Agent Middleware를 활용한 최종 응답 검토
- 별도의 Judge 모델을 통한 응답 안전성 평가
- 근거가 부족하거나 부적절한 응답의 수정 및 차단

### 2.6 기타 도구
- 도서관 문의 답변 초안 작성
- 사용자 정보 조회 및 저장
- 대출 정보 관련 안내

## 3. 개발 환경 및 사용 기술

| 구분 | 기술 |
|---|---|
| 프로그래밍 언어 | Python 3.12 |
| AI 모델 | Google Gemini |
| AI 프레임워크 | LangChain |
| 에이전트 및 메모리 관리 | LangGraph |
| 데이터베이스 | SQLite |
| 환경변수 관리 | python-dotenv |
| 실행 환경 | Windows |

### 주요 라이브러리

- `langchain`
- `langgraph`
- `langchain-google-genai`
- `python-dotenv`

## 4. 프로젝트 구조

```text
library_agent_project/
│
├── library_agent_gemini_final.py
├── requirements.txt
├── README.md
├── .gitignore
└── library_memory.db
```

※ `library_memory.db`는 실행 과정에서 생성되는 로컬 데이터베이스이며 GitHub 저장소에는 포함하지 않는다.

## 5. 설치 및 실행 방법

### 5.1 프로젝트 다운로드

GitHub 저장소를 Clone하거나 ZIP 파일로 다운로드한다.

[GitHub 저장소](https://github.com/skimskim49/library_agent_project)

### 5.2 라이브러리 설치

프로젝트 폴더에서 다음 명령어를 실행한다.

```bash
pip install -r requirements.txt
```

### 5.3 환경변수 설정

프로젝트 루트 디렉터리에 `.env` 파일을 생성하고 Gemini 사용에 필요한 환경변수를 설정한다.

```env
GEMINI_API_KEY=YOUR_API_KEY
GOOGLE_GENAI_USE_VERTEXAI=true
```

※ 실제 API 키는 GitHub에 업로드하지 않는다. Google Cloud 및 Vertex AI 사용 권한과 인증 환경이 필요하다.

### 5.4 에이전트 실행

```bash
python library_agent_gemini_final.py
```

## 6. 구현상의 제한 사항

- 도서 검색은 코드에 등록된 11권의 샘플 도서 데이터를 대상으로 수행한다.
- 실제 도서관 데이터베이스와 연동되어 있지 않으므로 실시간 대출 현황 및 반납 예정일을 조회할 수 없다.
- 이메일 관련 기능은 답변 초안 작성만 지원하며 실제 이메일 발송 기능은 구현하지 않았다.
- 장기 기억은 로컬 SQLite 데이터베이스를 기반으로 관리한다.
- 입력 분류 및 최종 응답 검토는 Gemini 모델을 활용하므로 모든 상황에서 완벽한 보안이나 응답 정확성을 보장하지 않는다.

## 7. 실행 시연 영상

본 프로젝트의 주요 기능 및 테스트 과정은 아래 YouTube 영상을 통해 확인할 수 있다.

- 시연 영상: (https://youtu.be/J6uIbJh7z8k?si=vlz1VntAg7cgWzG4)

## 8. GitHub 저장소

[AI Librarian Agent - GitHub](https://github.com/skimskim49/library_agent_project)

---

**프로젝트 목적:** LangChain 기반 싱글 에이전트의 Tool Calling, Runtime Context, Memory, Guardrail 및 최종 응답 검토 기능을 구현하고 테스트한다.