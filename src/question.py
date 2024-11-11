import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain_community.llms import Tongyi

from langchain_openai import ChatOpenAI
import tkinter as tk
from tkinter import scrolledtext,font
import threading

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

# print("\nRunning AI...")

# result = qa_chain.invoke({"query": question})
# print(result["result"])

class ChatBotUI:
    def __init__(self, master):
        self.master = master
        self.master.title("炒股养家AI")
        # 设置字体样式
        self.text_font = font.Font(family="Arial", size=12)  # 聊天区域字体
        self.entry_font = font.Font(family="Arial", size=12)  # 输入框字体
        
        self.chat_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, state='disabled', font=self.text_font)
        self.chat_area.pack(padx=10, pady=10)

        self.entry = tk.Entry(master, width=80, font=self.entry_font)
        self.entry.pack(padx=10, pady=10)
        self.entry.bind("<Return>", self.get_response)

        self.send_button = tk.Button(master, text="发送", command=self.get_response)
        self.send_button.pack(padx=10, pady=10)

        self.thinking_label = tk.Label(master, text="", fg="blue")  # 添加一个标签显示“思考中”
        self.thinking_label.pack(pady=5)

    def get_response(self, event=None):
        user_input = self.entry.get()
        self.update_chat_area("用户: " + user_input)
        self.thinking_label.config(text="思考中...")  # 显示“思考中”

        threading.Thread(target=self.call_ai_model, args=(user_input,)).start()  # 使用线程调用AI模型

        self.entry.delete(0, tk.END)  # 清空输入框

    def call_ai_model(self, question):
        result = qa_chain.invoke({"query": question})
        ai_response = result["result"]

        self.master.after(0, self.update_chat_area, "AI: " + ai_response)  # 更新聊天区域
        self.master.after(0, self.thinking_label.config, {'text': ""})  # 清空“思考中”

    def update_chat_area(self, message):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, message + '\n')
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END)  # 自动滚动到最后一行
        
        
if __name__ == "__main__":
    root = tk.Tk()
    chatbot = ChatBotUI(root)
    root.mainloop()