import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain_community.llms import Tongyi

from langchain_openai import ChatOpenAI


embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vector_db = Chroma(persist_directory="./chroma_db_test1", embedding_function=embedding_function)

question = "怎么判断赚钱效应和亏钱效应，赚钱效应好时应该怎么操作，亏钱效应应该怎么操作?"
print("\n查找知识库相似知识点:", question)

search_results = vector_db.similarity_search(question, k=2)

search_results_string = ""
for result in search_results:
    search_results_string += result.page_content + "\n\n"

print(search_results_string)

# llm = ChatOpenAI(temperature=0.0, base_url="http://localhost:1234/v1", api_key="not-needed")

#tongyi
os.environ["DASHSCOPE_API_KEY"] = "sk-*****************"
llm_ty = Tongyi()
# Build prompt
from langchain.prompts import PromptTemplate
template = """使用你的只是，结合以下内容回答问题。 \
    如果你不知道答案，就说你不知道，不要试图编造答案。 \
    。回答开头可以说"养家老师认为:" \
    {context} \
    问题: {question}
    答案:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm_ty,
                                        retriever=vector_db.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

print("\nRunning AI...")

result = qa_chain.invoke({"query": question})
print(result["result"])
