from langchain_core.prompts import ChatPromptTemplate

# The system prompt enforcing strict context adherence and specific fallback message.
SYSTEM_PROMPT = """You are a helpful AI Research Assistant.

Answer ONLY from the provided document context. Do not use outside knowledge.

If the answer cannot be found in the provided context, you MUST respond with exactly:
"I couldn't find that information in the uploaded documents."

Do not hallucinate or make up any answers.

Retrieved Context:
{context}"""

def get_qa_prompt() -> ChatPromptTemplate:
    """Get the RAG prompt template.

    Returns:
        ChatPromptTemplate: The prompt template for the RAG chain.
    """
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
