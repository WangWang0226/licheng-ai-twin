from dotenv import load_dotenv

load_dotenv()

from typing import Any, Dict, List

from langchain import hub
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.chat import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME
from callbacks import AgentCallbackHandler

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(
        model_name="gpt-4o-mini", 
        verbose=True, 
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Get original messages（A list）
    original_messages = retrieval_qa_chat_prompt.messages

    # Add a new system prompt
    new_system_prompt = SystemMessagePromptTemplate.from_template(
        "You are Licheng Wang. Answer the following user query strictly from the perspective of Licheng Wang. The response should fully align with the worldview, knowledge, morals, and values specific to Licheng Wang."
    )

    # Insert this prompt to the front of `messages`
    updated_messages = [new_system_prompt] + original_messages

    # Create a new ChatPromptTemplate including the new messages
    retrieval_qa_chat_prompt = ChatPromptTemplate(messages=updated_messages)
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    run_llm(query="who is licheng?", chat_history=[])
