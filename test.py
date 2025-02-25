from langchain import hub

retrieval_qa_chat = hub.pull("langchain-ai/retrieval-qa-chat")
print("*"*10)
print(retrieval_qa_chat)
print("*" * 10)
