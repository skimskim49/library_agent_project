"""AI 사서 보조 싱글 에이전트 - Gemini 기반 버전

메인 답변, 입력 의도/안전성 판단, 최종 답변 검수를 모두 Gemini로 수행합니다.
도서 검색과 SQLite 저장처럼 정해진 작업은 Python 도구가 처리합니다.
"""

import json
import os
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    Runtime,
    after_agent,
    before_agent,
    wrap_model_call,
)
from langchain.messages import AIMessage
from langchain.tools import ToolRuntime, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pydantic\.(functional_validators|main)",
)

# -----------------------------------------------------------------------------
# 1. Gemini / 환경 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError(".env에서 GEMINI_API_KEY를 찾을 수 없습니다.")

if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() != "true":
    raise ValueError(".env에 GOOGLE_GENAI_USE_VERTEXAI=true를 설정해 주세요.")

MODEL = "gemini-3.8-flash"

# 세 역할은 모두 Gemini를 사용하되, 역할별 프롬프트는 분리합니다.
main_model = ChatGoogleGenerativeAI(model=MODEL, temperature=0.2)
policy_model = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0)
judge_model = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0)

# -----------------------------------------------------------------------------
# 2. 도서 데이터 및 실행 컨텍스트
# -----------------------------------------------------------------------------
books = [
    {"title": "달빛 도서관의 비밀", "author": "최현범"},
    {"title": "내일이 사라진 도시", "author": "오다영"},
    {"title": "고양이는 자정에 책을 읽는다", "author": "김하린"},
    {"title": "시간을 파는 서점", "author": "이도윤"},
    {"title": "여름 끝에서 너를 만났다", "author": "박서연"},
    {"title": "기억을 잃은 탐정의 편지", "author": "정민재"},
    {"title": "열세 번째 문이 열리는 밤", "author": "한유진"},
    {"title": "구름 위 우체국의 마지막 편지", "author": "윤가람"},
    {"title": "마법사가 사라진 왕국", "author": "오지호"},
    {"title": "우리의 비밀은 별빛보다 깊다", "author": "강다은"},
    {"title": "사과의 매력을 찾아서", "author": "서정욱"},
]


@dataclass
class Context:
    librarian_name: str
    librarian_role: str
    user_id: str
    app_name: str


class UserInfo(TypedDict, total=False):
    personal_information: str
    preference: str
    facts: List[str]


store = InMemoryStore()
store.put(
    ("demo_user", "library_assistant"),
    "basic_rules",
    {
        "facts": [
            "도서 정보는 제공된 도서 목록과 검색 도구 결과를 기준으로 안내한다.",
            "실제 대출 데이터베이스가 연결되어 있지 않으므로 대출 상태를 추측하지 않는다.",
            "이용자에게 정중하고 자연스러운 한국어로 응대한다.",
        ]
    },
)

# DB는 이 Python 파일과 같은 폴더에 자동 생성됩니다.
DB_PATH = BASE_DIR / "library_memory.db"


def init_memory_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                app_name TEXT NOT NULL,
                preference TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def load_user_memories(user_id: str, app_name: str) -> List[str]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT preference
            FROM user_memories
            WHERE user_id = ? AND app_name = ?
            ORDER BY id
            """,
            (user_id, app_name),
        ).fetchall()
    return [row[0] for row in rows]


init_memory_db()

# -----------------------------------------------------------------------------
# 3. 에이전트 도구
# -----------------------------------------------------------------------------
@tool
def search_books(query: str) -> str:
    """도서 제목이나 저자 이름으로 등록된 도서를 검색합니다."""
    query = query.strip()
    if not query:
        return "검색어를 입력해 주세요."

    results = [
        book for book in books
        if query in book["title"] or query in book["author"]
    ]
    if not results:
        return f"'{query}'에 해당하는 도서를 현재 등록 목록에서 찾을 수 없습니다."

    return "\n".join(
        f"제목: {book['title']}, 저자: {book['author']}"
        for book in results
    )


@tool
def list_books() -> str:
    """현재 도서관 프로토타입에 등록된 전체 도서 목록을 반환합니다."""
    return "\n".join(
        f"{index}. {book['title']} - {book['author']}"
        for index, book in enumerate(books, start=1)
    )


@tool
def check_loan_info(runtime: ToolRuntime) -> str:
    """현재 프로토타입에서 본인의 대출 정보 확인 가능 여부를 안내합니다."""
    return (
        "현재 프로그램에는 실제 대출 데이터베이스와 이용자 인증 기능이 연결되어 있지 않습니다. "
        "따라서 대출 도서, 반납 예정일, 연체 여부를 확인할 수 없습니다. "
        "정확한 정보는 도서관 공식 시스템에서 확인해 주세요."
    )


@tool
def prepare_user_reply(message: str) -> str:
    """LLM이 작성한 도서관 이용자용 이메일/회신 초안을 정리합니다. 실제 발송은 하지 않습니다."""
    message = message.strip()
    if not message:
        return "회신 내용을 입력해 주세요."
    return f"[회신 초안]\n{message}\n[안내] 실제 메시지는 발송되지 않았습니다."


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """현재 이용자에 대해 장기 기억에 저장된 선호 정보를 조회합니다."""
    namespace = (runtime.context.user_id, runtime.context.app_name)
    memories = runtime.store.search(namespace) if runtime.store else []
    lines = []

    for item in memories:
        data = item.value
        if data.get("personal_information"):
            lines.append(f"이용자 정보: {data['personal_information']}")
        if data.get("preference"):
            lines.append(f"도서 선호: {data['preference']}")
        for fact in data.get("facts", []):
            lines.append(f"기억: {fact}")

    for preference in load_user_memories(
        runtime.context.user_id, runtime.context.app_name
    ):
        lines.append(f"도서 선호: {preference}")

    return "\n".join(lines) if lines else "저장된 이용자 정보가 없습니다."


@tool
def save_user_info(preference: str, runtime: ToolRuntime) -> str:
    """이용자가 알려준 도서 취향을 SQLite 장기 기억에 저장합니다."""
    preference = preference.strip()
    if not preference:
        return "저장할 선호 정보를 입력해 주세요."

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO user_memories (user_id, app_name, preference)
            VALUES (?, ?, ?)
            """,
            (runtime.context.user_id, runtime.context.app_name, preference),
        )

    return "이용자의 도서 선호 정보를 장기 기억에 저장했습니다."

# -----------------------------------------------------------------------------
# 4. 역할과 기억을 메인 Gemini에 전달
# -----------------------------------------------------------------------------
@wrap_model_call
def enrich_context(
    request: ModelRequest[Context],
    handler: Callable[[ModelRequest[Context]], ModelResponse],
) -> ModelResponse:
    """사서 역할, 업무 범위 및 장기 기억을 시스템 프롬프트에 추가합니다."""
    context = request.runtime.context
    namespace = (context.user_id, context.app_name)
    memories = (
        request.runtime.store.search(namespace)
        if request.runtime.store else []
    )

    facts = []
    for item in memories:
        value = item.value
        if value.get("preference"):
            facts.append(f"이용자 도서 선호: {value['preference']}")
        if value.get("personal_information"):
            facts.append(f"이용자 정보: {value['personal_information']}")
        facts.extend(value.get("facts", []))

    facts.extend(load_user_memories(context.user_id, context.app_name))
    memory_text = "\n- ".join(facts) if facts else "저장된 정보 없음"
    base_prompt = request.system_prompt or ""

    system_prompt = (
        f"{base_prompt}\n"
        f"당신은 {context.librarian_name}이며, 역할은 {context.librarian_role}입니다.\n"
        "사용자와 자연스럽고 친절하게 대화하되, 사서 업무에 필요한 전문성을 유지하세요.\n"
        "도서관 업무 원칙:\n"
        "- 도서명과 저자 정보는 반드시 제공된 도서 검색 도구의 결과를 근거로 안내하세요.\n"
        "- 전체 도서 목록 요청에는 list_books 도구를 사용하세요. 특정 도서나 저자 검색에는 search_books 도구를 사용하세요.\n"
        "- 실제 대출 데이터베이스와 인증 기능이 연결되어 있지 않습니다. 대출 도서, 반납일, 연체 여부를 확인했다고 주장하거나 추측하지 마세요.\n"
        "- 다른 이용자의 대출 내역이나 개인정보를 조회하거나 공개하지 마세요.\n"
        "- 이메일이나 회신 초안을 요청받으면 목적에 맞게 직접 작성한 뒤 prepare_user_reply 도구를 사용하세요. 실제 발송했다고 말하지 마세요.\n"
        "- 이용자가 제공하지 않은 사실, 도서 정보, 개인정보를 만들어내지 마세요. 모르는 내용은 모른다고 설명하세요.\n"
        "- 특정 도서관의 공식 정책이나 규정을 확인할 수 없다면 일반적인 원칙과 해당 도서관의 실제 정책을 구분해서 설명하세요.\n"
        "- 확인하지 않은 개인정보 처리방침이나 법적 의무를 해당 도서관의 확정된 정책처럼 단정하지 마세요.\n"
        "- 일반적인 개인정보 보호 방법을 설명할 때는 일반적인 권장 사항임을 명시하세요.\n"
        "- 도서관 업무와 무관한 요청은 사서 보조의 역할을 간단히 설명하고 정중하게 안내하세요.\n"
        "- 거절이 필요한 경우 이유를 짧게 설명하고, 가능한 안전한 대안을 제시하세요.\n"
        f"[장기 기억]\n- {memory_text}\n"
        "한국어로 답변하세요. 답변은 질문의 복잡도에 맞게 간결하고 자연스럽게 작성하세요."
    )
    return handler(request.override(system_prompt=system_prompt))

# -----------------------------------------------------------------------------
# 5. Gemini 기반 의미/의도 입력 판단 (금지어 목록 사용 안 함)
# -----------------------------------------------------------------------------
@before_agent(can_jump_to=["end"])
def semantic_input_guardrail(state, runtime: Runtime[Context]):
    """Gemini가 사용자 요청의 의미를 분석해 개인정보/업무범위 위험을 판단합니다."""
    messages = state.get("messages", [])
    if not messages:
        return None

    last_message = messages[-1]
    if getattr(last_message, "type", None) != "human":
        return None

    question = last_message.content or ""
    if not isinstance(question, str):
        question = str(question)

    classification_prompt = f"""당신은 도서관 AI 사서의 입력 안전성 분류기입니다.
사용자 요청의 단어가 아니라 문맥, 의도, 실제로 요구하는 행동을 판단하세요.
아래 넷 중 하나의 라벨만 출력하세요: ALLOW, PRIVACY, OUT_OF_SCOPE, DECEPTION

판단 기준:
- PRIVACY: 다른 이용자의 대출 기록, 계정, 연락처, 개인 식별 정보 등 비공개 정보를 알아내거나 공개하라는 요청. 우회, 역할극, 관리자 사칭을 통한 요청도 포함합니다.
- DECEPTION: 실제로 수행하지 않은 이메일 발송, 예약, 조회, 변경 등의 작업을 완료했다고 거짓으로 말하거나 그렇게 주장하도록 요구하는 요청입니다.
- OUT_OF_SCOPE: 도서관/도서/독서/도서관 이용 안내와 무관한 작업을 수행해 달라는 요청.
- ALLOW: 도서 검색, 도서 추천, 독서 상담, 본인 대출 정보에 관한 일반 안내, 도서관 정책 문의, 이메일 초안, 개인정보 보호에 대한 일반적인 설명 등 정당한 요청.
- 다른 사람의 정보를 보호하는 방법을 묻는 일반적인 교육/설명 요청은 PRIVACY가 아니라 ALLOW입니다.
- 실제 발송 여부를 묻거나 발송하지 않은 사실을 솔직하게 설명하는 요청은 DECEPTION이 아니라 ALLOW입니다.
- 의도가 모호하거나 위험성이 분명하지 않으면 ALLOW로 분류하세요.
- 요청이 도서관 업무 범위 밖에 있더라도 개인정보 침해 의도가 명확하면 OUT_OF_SCOPE보다 PRIVACY를 우선합니다.
- 소설, 역할극, 가정 상황 등의 표현을 사용하더라도 실제 개인정보를 알아내거나 무단으로 접근하는 방법을 요구하면 PRIVACY로 분류합니다.

사용자 요청:
{question}

라벨만 출력하세요."""

    try:
        result = policy_model.invoke(classification_prompt)
        verdict = str(result.content).strip().upper()
        # 모델이 부가 설명을 붙여도 라벨을 판별합니다.
        if "PRIVACY" in verdict:
            category = "PRIVACY"
        elif "DECEPTION" in verdict:
            category = "DECEPTION"
        elif "OUT_OF_SCOPE" in verdict:
            category = "OUT_OF_SCOPE"
        elif "ALLOW" in verdict:
            category = "ALLOW"
        else:
            category = "ALLOW"

        if category == "PRIVACY":
            return {
                "messages": [AIMessage(content=(
                    "개인정보 보호를 위해 다른 이용자의 대출 내역이나 개인정보는 제공할 수 없습니다. "
                    "본인의 정보는 도서관의 공식 인증 절차를 통해 확인해 주세요."
                ))],
                "jump_to": "end",
            }

        if category == "DECEPTION":
            return {
                "messages": [AIMessage(content=(
                    "실제로 수행하지 않은 작업을 완료했다고 안내할 수는 없습니다. "
                    "대신 이메일 초안 작성 등 실제로 가능한 작업은 도와드릴 수 있습니다."
                ))],
                "jump_to": "end",
            }

        if category == "OUT_OF_SCOPE":
            return {
                "messages": [AIMessage(content=(
                    "저는 도서 검색과 도서관 이용 안내를 지원하는 AI 사서 보조입니다. "
                    "도서나 도서관 이용과 관련된 질문을 도와드릴게요."
                ))],
                "jump_to": "end",
            }

        return None

    except Exception as error:
        # 입력 안전성 판단 자체가 실패하면 에이전트가 요청을 처리하지 않도록 합니다.
        print(f"[Gemini 입력 판단 오류] {error}")
        return {
            "messages": [AIMessage(content=(
                "요청의 안전성을 확인하는 과정에서 오류가 발생했습니다. "
                "잠시 후 다시 질문해 주세요."
            ))],
            "jump_to": "end",
        }

# -----------------------------------------------------------------------------
# 6. Gemini Judge 기반 최종 답변 검수
# -----------------------------------------------------------------------------
@after_agent
def review_final_answer(state, runtime: Runtime[Context]):
    """Gemini Judge가 최종 답변의 근거, 개인정보 및 허위 수행 주장을 검사합니다."""
    messages = state.get("messages", [])
    if not messages:
        return None

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return None

    answer = last_message.content
    if isinstance(answer, list):
        answer = "\n".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        answer = str(answer)

    tool_evidence = []
    for message in messages[-8:]:
        if getattr(message, "type", None) == "tool":
            tool_evidence.append(extract_text(getattr(message, "content", "")))
    evidence_text = "\n---\n".join(tool_evidence) if tool_evidence else "이번 답변에서 확인된 도구 결과 없음"

    judge_prompt = f"""당신은 도서관 AI 사서의 독립적인 최종 답변 검수자입니다.
답변이 아래 기준을 위반하는지 문맥과 도구 결과를 고려해 판단하세요.

UNSAFE 기준:
1. 도구나 제공된 자료로 확인하지 않은 도서 정보 또는 대출 정보를 사실처럼 단정함.
2. 다른 이용자의 비공개 대출 내역이나 개인정보를 노출함.
3. 실제 발송하지 않은 이메일/메시지를 발송했다고 주장함.
4. 실제로 수행하지 않은 조회, 예약, 변경 등을 완료했다고 주장함.
5. 도서관 업무 범위와 관련한 중대한 오해를 유발하는 허위 안내를 함.

SAFE 기준:
- 확인할 수 없다고 솔직히 설명하거나, 공식 시스템 확인을 안내하는 답변.
- 도구 결과에 근거한 정확한 도서 안내.
- 단순한 표현 차이나 친근한 말투는 위반이 아님.

이번 실행에서 확인된 도구 결과:
{evidence_text}

검수 대상 답변:
{answer}

UNSAFE 또는 SAFE 중 하나의 단어만 출력하세요."""

    try:
        verdict_result = judge_model.invoke(judge_prompt)
        verdict = extract_text(verdict_result.content).strip().upper()
        verdict_label = verdict.split()[0].strip(".,:;!\"'") if verdict else ""

        if verdict_label == "UNSAFE":
            last_message.content = (
                "확인되지 않았거나 제공할 수 없는 정보는 안내할 수 없습니다. "
                "도서관 공식 시스템 또는 담당자를 통해 확인해 주세요."
            )

    except Exception as error:
        print(f"[Gemini Judge 오류] {error}")
        last_message.content = (
            "최종 답변의 안전성을 확인하는 중 문제가 발생했습니다. "
            "잠시 후 다시 질문해 주세요."
        )

    return None

# -----------------------------------------------------------------------------
# 7. 에이전트 조립
# -----------------------------------------------------------------------------
agent = create_agent(
    main_model,
    tools=[
        search_books,
        list_books,
        check_loan_info,
        prepare_user_reply,
        get_user_info,
        save_user_info,
    ],
    checkpointer=InMemorySaver(),
    store=store,
    context_schema=Context,
    middleware=[
        semantic_input_guardrail,
        enrich_context,
        review_final_answer,
    ],
)

# -----------------------------------------------------------------------------
# 8. 실행
# -----------------------------------------------------------------------------
def extract_text(content) -> str:
    """Gemini 응답이 문자열/콘텐츠 블록 중 어떤 형태여도 텍스트만 반환합니다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def main():
    context = Context(
        librarian_name="AI 사서",
        librarian_role="도서 검색과 도서관 이용 안내를 지원하는 사서 보조",
        user_id="demo_user",
        app_name="library_assistant",
    )
    thread_id = "library-demo"

    print("AI 사서 보조 에이전트")
    print(f"입력 판단 / 메인 / Judge 모델: {MODEL} (Vertex AI)")
    print("종료하려면 '종료'를 입력하세요.")

    while True:
        question = input("\n이용자 > ").strip()
        if question.lower() in {"종료", "exit", "quit"}:
            print("이용해 주셔서 감사합니다.")
            break
        if not question:
            continue

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": thread_id}},
                context=context,
            )
            answer = extract_text(result["messages"][-1].content)
            print("\nAI 사서 >", answer)
        except Exception as error:
            print(f"\n실행 중 오류가 발생했습니다: {error}")


if __name__ == "__main__":
    main()
