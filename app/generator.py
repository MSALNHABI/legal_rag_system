from pathlib import Path
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LegalAnswerGenerator:
    """
    Grounded legal answer generator.

    Purpose:
    - Use retrieved legal chunks as the only source of truth.
    - Generate professional Arabic legal answers.
    - Attach citation markers to claims.
    - Decline out-of-scope questions.
    """

    def __init__(self) -> None:
        load_dotenv(PROJECT_ROOT / ".env")

        self.openai_api_key = self._get_env_value("OPENAI_API_KEY")
        self.chat_model = self._get_env_value("OPENAI_CHAT_MODEL", "gpt-5.4-mini")

        self.llm = ChatOpenAI(
            model=self.chat_model,
            api_key=self.openai_api_key,
        )

    def _get_env_value(self, name: str, default: str | None = None) -> str:
        value = os.getenv(name, default)

        if value is None or not value.strip():
            raise ValueError(f"Missing required environment variable: {name}")

        return value.strip()

    def _format_context(self, retrieved_chunks: list[dict[str, Any]]) -> str:
        context_blocks = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            citation_label = f"C{index}"

            block = f"""
[{citation_label}]
chunk_id: {chunk["chunk_id"]}
law_name_ar: {chunk["law_name_ar"]}
law_name_en: {chunk["law_name_en"]}
article_number: {chunk["article_number"]}
article_heading_ar: {chunk["article_heading_ar"]}
text:
{chunk["text"]}
""".strip()

            context_blocks.append(block)

        return "\n\n" + ("\n" + "-" * 80 + "\n").join(context_blocks)

    def _build_system_prompt(self) -> str:
        return """
أنت مساعد قانوني متخصص في الإجابة من نظام العمل ونظام التأمينات الاجتماعية في المملكة العربية السعودية.

القواعد الإلزامية:
1. أجب فقط من السياق القانوني المقدم لك.
2. لا تستخدم معرفة خارجية.
3. لا تخترع مواد أو أحكامًا غير موجودة في السياق.
4. إذا لم تجد الجواب في السياق المسترجع، قل فقط: "لا أستطيع الإجابة من النصوص القانونية المسترجعة المتاحة."
5. كل حكم قانوني أو معلومة مهمة يجب أن تحتوي على citation مثل [C1] أو [C2].
6. لا تذكر chunk_id داخل نص الإجابة، استخدم فقط citation labels مثل [C1].
7. اجعل الإجابة عربية، مهنية، واضحة، ومختصرة.
8. إذا كان السؤال يطلب شرح مادة محددة، اشرحها بلغة مبسطة دون الخروج عن النص.
9. إذا وُجدت أكثر من مادة مرتبطة، اذكر الفرق بينها فقط عند الحاجة.
10. لا تقدم استشارة قانونية شخصية نهائية، بل اشرح الحكم النظامي من النصوص المتاحة.
11. لا تضف عبارات متابعة أو عروضًا إضافية مثل: "إذا رغبت" أو "أستطيع أيضًا".
12. لا تستخدم citation إلا إذا كان النص cited يدعم الجملة مباشرة.
""".strip()

    def _build_user_prompt(
        self,
        question: str,
        context: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        history_text = ""

        if chat_history:
            history_items = []

            for item in chat_history[-6:]:
                role = item.get("role", "")
                content = item.get("content", "")

                if role and content:
                    history_items.append(f"{role}: {content}")

            if history_items:
                history_text = "\n\nسياق المحادثة السابقة:\n" + "\n".join(history_items)

        return f"""
السؤال:
{question}

{history_text}

السياق القانوني المسترجع:
{context}

المطلوب:
اكتب إجابة قانونية عربية دقيقة ومختصرة.
يجب وضع citation بعد كل حكم قانوني مهم.
لا تضف أي معلومات غير موجودة في السياق.
لا تضف أسئلة متابعة أو عروضًا إضافية في نهاية الإجابة.
""".strip()

    def _extract_used_citation_numbers(self, answer: str) -> list[int]:
        matches = re.findall(r"\[C(\d+)\]", answer)

        citation_numbers = sorted(
            {
                int(match)
                for match in matches
                if match.isdigit()
            }
        )

        return citation_numbers

    def _build_citations(
        self,
        retrieved_chunks: list[dict[str, Any]],
        used_citation_numbers: list[int],
    ) -> list[dict[str, Any]]:
        citations = []

        for citation_number in used_citation_numbers:
            index = citation_number - 1

            if index < 0 or index >= len(retrieved_chunks):
                continue

            chunk = retrieved_chunks[index]

            citations.append(
                {
                    "label": f"C{citation_number}",
                    "chunk_id": chunk["chunk_id"],
                    "law_id": chunk["law_id"],
                    "law_name_ar": chunk["law_name_ar"],
                    "law_name_en": chunk["law_name_en"],
                    "article_number": chunk["article_number"],
                    "article_heading_ar": chunk["article_heading_ar"],
                    "source_file": chunk["source_file"],
                    "text_preview": chunk["text"][:500],
                }
            )

        return citations

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        question = question.strip()

        if not question:
            return {
                "answer": "الرجاء إدخال سؤال قانوني واضح.",
                "citations": [],
                "used_citation_labels": [],
                "model": self.chat_model,
                "grounded": False,
            }

        if not retrieved_chunks:
            return {
                "answer": "لا أستطيع الإجابة من النصوص القانونية المسترجعة المتاحة.",
                "citations": [],
                "used_citation_labels": [],
                "model": self.chat_model,
                "grounded": True,
            }

        context = self._format_context(retrieved_chunks)

        messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(
                content=self._build_user_prompt(
                    question=question,
                    context=context,
                    chat_history=chat_history,
                )
            ),
        ]

        response = self.llm.invoke(messages)
        answer = str(response.content).strip()

        refusal_phrases = [
            "لا أستطيع الإجابة",
            "لا تتضمن النصوص",
            "غير موجود في السياق",
            "غير موجودة في السياق",
            "غير متاح في السياق",
        ]

        is_refusal = any(phrase in answer for phrase in refusal_phrases)

        used_citation_numbers = self._extract_used_citation_numbers(answer)

        if not used_citation_numbers and not is_refusal:
            answer = f"{answer}\n\nالمصدر: [C1]"
            used_citation_numbers = [1]

        citations = self._build_citations(
            retrieved_chunks=retrieved_chunks,
            used_citation_numbers=used_citation_numbers,
        )

        return {
            "answer": answer,
            "citations": citations,
            "used_citation_labels": [f"C{number}" for number in used_citation_numbers],
            "model": self.chat_model,
            "grounded": bool(citations) or is_refusal,
        }