from uuid import uuid4


class ConversationMemory:
    """
    Simple in-memory conversation store.

    Purpose:
    - Keep short chat history for each conversation_id.
    - This is enough for assignment/demo.
    - Later it can be replaced with Redis, SQLite, or database storage.
    """

    def __init__(self, max_messages_per_conversation: int = 12) -> None:
        self.max_messages_per_conversation = max_messages_per_conversation
        self.store: dict[str, list[dict[str, str]]] = {}

    def create_conversation_id(self) -> str:
        conversation_id = str(uuid4())
        self.store[conversation_id] = []
        return conversation_id

    def get_or_create_conversation(self, conversation_id: str | None) -> str:
        if conversation_id and conversation_id in self.store:
            return conversation_id

        if conversation_id and conversation_id not in self.store:
            self.store[conversation_id] = []
            return conversation_id

        return self.create_conversation_id()

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        return self.store.get(conversation_id, [])

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        if conversation_id not in self.store:
            self.store[conversation_id] = []

        self.store[conversation_id].append(
            {
                "role": role,
                "content": content,
            }
        )

        self.store[conversation_id] = self.store[conversation_id][
            -self.max_messages_per_conversation:
        ]