from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()

# Tool for answering from an pdf provided by the user
@tool
def pdf_ans(pdf_path: str, user_query: str) -> str:
  '''Provides answer by referencing pdf'''
  loader = PyPDFLoader(pdf_path)
  pages = loader.load_and_split()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  documents = text_splitter.split_documents(pages)
  vector = FAISS.from_documents(documents, embeddings)
  prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

  <context>
  {context}
  </context>

  Question: {input}""")
  document_chain = create_stuff_documents_chain(llm, prompt)
  retriever = vector.as_retriever(search_kwargs={"k":5})
  retrieval_chain = create_retrieval_chain(retriever, document_chain)
  
  response = retrieval_chain.invoke({"input": f"{user_query}"})

  return response['answer']

# tool to provide responses based on the url provided by the user
@tool
def web_ans(url: str, user_query: str) -> str:
  '''Provides answer by referenced url by extracting the content'''
  loader = WebBaseLoader("https://python.langchain.com/docs/integrations/document_loaders/web_base/")
  data = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  documents = text_splitter.split_documents(data)
  vector = FAISS.from_documents(documents, embeddings)
  prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

  <context>
  {context}
  </context>

  Question: {input}""")
  document_chain = create_stuff_documents_chain(llm, prompt)
  retriever = vector.as_retriever(search_kwargs={"k":5})
  retrieval_chain = create_retrieval_chain(retriever, document_chain)
  
  response = retrieval_chain.invoke({"input": f"{user_query}"})

  return response['answer']


from youtube_transcript_api import YouTubeTranscriptApi
def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en','ja','hi',])
        text = ' '.join([t['text'] for t in transcript])
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# tool to provide responses based on the youtube url or video_id provided by the user
@tool
def yt_ans(video_id: str, user_query: str) -> str:
  """"Provides response based on the transcript of youtube video_id provided. video_id is usually an alphanumeric sequence"""
  transcript_text = get_youtube_transcript(video_id)

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  chunks = text_splitter.split_text(transcript_text)
  chunks = text_splitter.create_documents(chunks)
  vector = FAISS.from_documents(chunks, embeddings)


  prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
  <context>
  {context}
  </context>

  Question: {input}""")

  document_chain = create_stuff_documents_chain(llm, prompt)
  retriever = vector.as_retriever(search_kwargs={"k":3})
  retrieval_chain = create_retrieval_chain(retriever, document_chain)

  response = retrieval_chain.invoke({"input": f"{user_query}"})

  return response['answer']

