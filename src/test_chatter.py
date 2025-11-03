from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS

vectorstore = FAISS.load_local("../data/papers_index/faiss_index")
retriever = vectorstore.as_retriever()

llm = LlamaCpp(model_path="../models/your_local_model.bin")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

query = "What does research say about sponsor role clarity in addiction recovery?"
answer = qa_chain.run(query)
print(answer)
