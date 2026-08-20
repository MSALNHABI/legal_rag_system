from pathlib import Path
import os

import requests
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSS_PATH = PROJECT_ROOT / "frontend" / "assets" / "style.css"

load_dotenv(PROJECT_ROOT / ".env")

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title="Saudi Legal RAG",
    page_icon="⚖️",
    layout="wide",
)


def load_css() -> None:
    """Load the external CSS theme."""

    if not CSS_PATH.exists():
        st.warning(f"CSS file not found: {CSS_PATH}")
        return

    css = CSS_PATH.read_text(encoding="utf-8")

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Initialize the current conversation state."""

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


def call_chat_api(query: str) -> dict:
    """Send a query to the FastAPI backend."""

    payload = {
        "query": query,
        "conversation_id": st.session_state.conversation_id,
        "top_k": 10,
        "semantic_k": 20,
        "keyword_k": 20,
    }

    response = requests.post(
        f"{API_BASE_URL}/chat",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def render_sidebar() -> None:
    """Render the sidebar."""

    with st.sidebar:
        with st.container(
            key="sidebar_brand",
            border=True,
        ):
            st.markdown("### Saudi Legal RAG ⚖️")
            st.markdown(
                "واجهة قانونية للاستعلام من نظام العمل "
                "ونظام التأمينات الاجتماعية."
            )

        if st.button(
            "محادثة جديدة",
            key="new_conversation",
            use_container_width=True,
        ):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()


def render_header() -> None:
    """Render the centered platform header."""

    with st.container(
        key="hero_section",
        border=True,
    ):
        st.caption("منصة قانونية عربية موثقة")

        st.title("Saudi Legal RAG ⚖️")

        st.markdown(
            """
نظام سؤال وجواب قانوني مبني على نظام العمل ونظام التأمينات الاجتماعية في السعودية،

مع إجابات مستندة إلى النصوص القانونية ومرفقة بالمراجع ذات الصلة.
"""
        )


def render_welcome_section() -> str | None:
    """Render suggested questions before the conversation starts."""

    if st.session_state.messages:
        return None

    with st.container(key="welcome_heading"):
        st.subheader("ابدأ بأسئلة شائعة")

    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )

    with col1:
        with st.container(
            key="end_service_card",
            border=True,
        ):
            st.markdown("#### مكافأة نهاية الخدمة")
            st.markdown(
                "اسأل عن استحقاق المكافأة، وطريقة الحساب، "
                "وأثر الاستقالة على مقدارها."
            )

        ask_end_of_service = st.button(
            "اسأل عن مكافأة نهاية الخدمة",
            key="ask_end_of_service",
            use_container_width=True,
        )

    with col2:
        with st.container(
            key="probation_card",
            border=True,
        ):
            st.markdown("#### فترة التجربة وفسخ العقد")
            st.markdown(
                "استفسر عن فترة التجربة، والإنهاء خلالها، "
                "والحالات التي يجوز فيها فسخ العقد."
            )

        ask_probation = st.button(
            "اسأل عن فترة التجربة",
            key="ask_probation",
            use_container_width=True,
        )

    with col3:
        with st.container(
            key="insurance_card",
            border=True,
        ):
            st.markdown("#### اشتراكات التأمينات")
            st.markdown(
                "اسأل عن نسب الاشتراك، وفرع المعاشات، "
                "والأحكام المرتبطة بالتأمينات الاجتماعية."
            )

        ask_insurance = st.button(
            "اسأل عن اشتراكات التأمينات",
            key="ask_insurance",
            use_container_width=True,
        )

    if ask_end_of_service:
        return "متى يستحق العامل مكافأة نهاية الخدمة؟"

    if ask_probation:
        return "ما أحكام إنهاء عقد العمل خلال فترة التجربة؟"

    if ask_insurance:
        return "كم نسبة الاشتراك في فرع المعاشات؟"

    return None


def render_citations(
    citations: list[dict],
) -> None:
    """Render citations used in the answer."""

    if not citations:
        st.info(
            "لا توجد مراجع؛ غالبًا لأن السؤال خارج نطاق "
            "النصوص القانونية المسترجعة."
        )
        return

    for citation in citations:
        label = str(citation.get("label") or "")
        law_name_ar = str(citation.get("law_name_ar") or "")
        article_number = str(citation.get("article_number") or "")
        chunk_id = str(citation.get("chunk_id") or "")
        text_preview = str(citation.get("text_preview") or "")

        with st.container(border=True):
            st.markdown(
                f"**[{label}] {law_name_ar}**"
            )

            st.markdown(
                f"**المادة:** {article_number}"
            )

            st.code(
                f"chunk_id: {chunk_id}",
                language=None,
            )

            st.markdown(text_preview)


def render_retrieved_chunks(
    chunks: list[dict],
) -> None:
    """Render chunks returned by the retriever."""

    if not chunks:
        st.info("لا توجد مقاطع مسترجعة.")
        return

    for chunk in chunks:
        rank = str(chunk.get("rank") or "")
        law_name_ar = str(chunk.get("law_name_ar") or "")
        article_number = str(chunk.get("article_number") or "")
        chunk_id = str(chunk.get("chunk_id") or "")
        retrieval_method = str(chunk.get("retrieval_method") or "")
        semantic_rank = str(chunk.get("semantic_rank") or "")
        keyword_rank = str(chunk.get("keyword_rank") or "")
        text_preview = str(chunk.get("text_preview") or "")

        matched_by_values = chunk.get("matched_by") or []

        matched_by = ", ".join(
            str(value)
            for value in matched_by_values
        )

        with st.container(border=True):
            st.markdown(
                f"**الترتيب {rank}** | "
                f"{law_name_ar} | "
                f"المادة {article_number}"
            )

            st.code(
                "\n".join(
                    [
                        f"chunk_id: {chunk_id}",
                        f"method: {retrieval_method}",
                        f"semantic_rank: {semantic_rank}",
                        f"keyword_rank: {keyword_rank}",
                        f"matched_by: {matched_by}",
                    ]
                ),
                language=None,
            )

            st.markdown(text_preview)


def render_assistant_metadata(
    message: dict,
) -> None:
    """Render citations, chunks, and answer metadata."""

    citations = message.get("citations") or []
    retrieved_chunks = message.get("retrieved_chunks") or []

    grounded = message.get("grounded")
    model = message.get("model")
    latency_ms = message.get("latency_ms")

    with st.expander("المراجع القانونية"):
        render_citations(citations)

    with st.expander("المقاطع المسترجعة"):
        render_retrieved_chunks(
            retrieved_chunks
        )

    st.caption(
        f"Grounded: {grounded} | "
        f"Model: {model} | "
        f"Latency: {latency_ms} ms"
    )


def render_chat_history() -> None:
    """Render all messages in the current conversation."""

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant":
                render_assistant_metadata(
                    message
                )


def handle_query(query: str) -> None:
    """Process one user query."""

    clean_query = query.strip()

    if not clean_query:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": clean_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(clean_query)

    with st.chat_message("assistant"):
        with st.spinner(
            "جاري البحث في النصوص القانونية وتوليد الإجابة..."
        ):
            try:
                api_result = call_chat_api(
                    clean_query
                )

                st.session_state.conversation_id = (
                    api_result["conversation_id"]
                )

                assistant_message = {
                    "role": "assistant",
                    "content": api_result["answer"],
                    "citations": api_result.get(
                        "citations",
                        [],
                    ),
                    "retrieved_chunks": api_result.get(
                        "retrieved_chunks",
                        [],
                    ),
                    "grounded": api_result.get(
                        "grounded"
                    ),
                    "model": api_result.get(
                        "model"
                    ),
                    "latency_ms": api_result.get(
                        "latency_ms"
                    ),
                }

                st.session_state.messages.append(
                    assistant_message
                )

                st.markdown(
                    assistant_message["content"]
                )

                render_assistant_metadata(
                    assistant_message
                )

            except requests.RequestException as error:
                st.error(
                    "حدث خطأ أثناء الاتصال بالـ API. "
                    "تأكد أن FastAPI يعمل على "
                    "http://127.0.0.1:8000"
                )

                st.code(
                    str(error),
                    language=None,
                )


def main() -> None:
    """Run the Streamlit application."""

    load_css()
    initialize_session_state()

    render_sidebar()
    render_header()

    suggested_query = render_welcome_section()

    render_chat_history()

    typed_query = st.chat_input(
        "اكتب سؤالك القانوني هنا..."
    )

    query_to_process = (
        suggested_query
        or typed_query
    )

    if query_to_process:
        handle_query(query_to_process)


if __name__ == "__main__":
    main()