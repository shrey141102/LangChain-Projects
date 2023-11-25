import re
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import AsyncChromiumLoader, AsyncHtmlLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings, HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
import os
import pickle
from langchain.chains import LLMChain, ConversationChain, VectorDBQA
from PyPDF2 import PdfReader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.llms import HuggingFaceHub, GooglePalm
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
#from apikey import OPENAI_API, HUGGINGFACE_API, GOOGLE_PALM_API_KEY
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate, ChatPromptTemplate

repo_id = "google/flan-t5-xxl"

full_data = []

link_pattern = r'(https?://\S+)'
links = []
file_name = "data.txt"
exclude_domains = ['facebook.com', 'instagram.com', 'linkedin.com', 'github.com', 'youtube.com', 'twitter.com']

# if base_url[-1] == '/':
#     base_url = base_url[:-1]

total_html = ""

template = """
You are a friendly customer service representative chatbot for the company {name}. 
I will share a customer's message with you and you will give  the best answer that 
You should send to this customer based on the given company data, 
and you will follow ALL of the rules below:

1/ Response should be only on the basis of the data given to you below. It must be very similar accurate and short to the company details provided. It should not be very long. The response is only to be answered from the information given to you by us.

2/ If the data are irrelevant, then answer "I don't have information about this".

3/ Encourage the customer to ask follow up questions.

Below is a message I received from the customer:
{message}

Below is the chat history of our conversation:
{chat_history}

Here is a list of the information that we have related to the question:
{closest_match}

Please write the best response that I should send to this customer:
"""

prompt = PromptTemplate(
    input_variables=["name", "message", "chat_history", "closest_match"],
    template=template
)

# CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(template)
persist_directory = "chroma_db"

def extract_site_name(url):
    # Define a regex pattern to match the main part of the domain
    pattern = re.compile(r'https?://(?:www\.)?([a-zA-Z0-9-]+)\.[a-zA-Z]{2,}')

    # Use the regex pattern to search for matches in the URL
    match = pattern.match(url)

    if match:
        # The first group in the match contains the extracted site name
        site_name = match.group(1)
        return site_name
    else:
        return None


def choose_llm(name):
    if name == "ChatOpenAI":
        return ChatOpenAI(temperature=0.2)
    if name == "ChatGooglePalm":
        return ChatGooglePalm(temperature=0.8)
    if name == "HuggingFaceHub":
        return HuggingFaceHub(
                repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 256}
            )
def choose_embeddings(name):
    if name ==  "OpenAIEmbeddings":
        return OpenAIEmbeddings()
    if name ==  "SentenceTransformerEmbeddings":
        return SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    if name == "HuggingFaceEmbeddings":
        return HuggingFaceEmbeddings()


def starter_code(base_url):

    if base_url[-1] == '/':
        base_url = base_url[:-1]

    # Scrape the main url
    docs_transformed = scraper(base_url)

    with open(file_name, "w") as file:
        file.write(f'This is the data from {base_url}\n')
        file.write(docs_transformed[0].page_content)

    # Find nav links of baseURL
    paths = find_navlinks(docs_transformed)

    # Scrape data of the nav links
    for path in paths:
        docs_transformed = scraper(base_url+path)
        x = check_new_navlinks(docs_transformed, paths)
        if x is not None:
            paths.append(x)
        with open(file_name, "a") as file:
            file.write("\n")
            file.write("\n")
            file.write(f'This is the data from {base_url+path}\n')
            file.write(docs_transformed[0].page_content)
    print(paths)

    # After scraping everywhere we collect a list of links and remove some links such as twitter, insta etc.
    new_links = list(set(links))
    new_links = [link for link in new_links if not any(domain in link for domain in exclude_domains)]
    try:
        new_links.remove(base_url+'/')
        new_links.remove(base_url)
    except Exception as e:
        print()
    print(new_links)

    # Scrape data from those links also
    for path in new_links:
        docs_transformed = scraper(path)
        ## x = check_new_navlinks(docs_transformed, paths)
        ## if x is not None:
        ##     paths.append(x)
        with open(file_name, "a") as file:
            file.write("\n")
            file.write("\n")
            file.write(f'This is the data from {path}\n')
            file.write(docs_transformed[0].page_content)

    return True



def check_new_navlinks(data, path_main):
    paths = re.findall(r'\(/\w+\)', data[0].page_content)
    filtered_paths = [path[1:-1] for path in paths]
    unique_paths = list(set(filtered_paths))
    new_links = [word for word in unique_paths if word not in path_main]
    if len(new_links)>0:
        return new_links
    return

def scraper(url):
    loader = AsyncChromiumLoader([url])
    # loader = AsyncHtmlLoader(url)
    html = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p", "li", "div", "a"]) #, "span", "h", "link"


    #extract links
    x = [link.rstrip(')') for link in re.findall(link_pattern, docs_transformed[0].page_content)]
    for link in x:
        links.append(link)

    return docs_transformed

def find_navlinks(docs_transformed):
    paths = re.findall(r'\(/\w+\)', docs_transformed[0].page_content)
    filtered_paths = [path[1:-1] for path in paths]
    unique_paths = list(set(filtered_paths))
    return unique_paths


def memory_query(query, embeddings, llm):

    def retrieve_info(query):
        similar_response = vectorstore.similarity_search_with_score(query, k=3)
        page_contents_array = [doc[0].page_content for doc in similar_response]
        return page_contents_array

    loader = TextLoader(file_name)
    documents = loader.load()

    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
    # text_splitter =CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    # vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
    vectorstore = FAISS.from_documents(documents, embeddings)
    # vectorstore.persist()

    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True) #, memory_key="chat_history", return_messages=True

    chain = LLMChain(llm=llm, prompt=prompt)
    qa = ConversationalRetrievalChain.from_llm(llm,
                                               vectorstore.as_retriever(),
                                               memory=memory,
                                               # verbose=True,
                                               # condense_question_prompt=prompt,
                                               chain_type="stuff")



        # closest_match = retrieve_info(query)
        #
        # response = chain.run(name=extract_site_name(base_url), message=query, chat_history=memory.load_memory_variables({}), closest_match=closest_match)
        # print(response)
        #
        # memory.save_context({"input": query}, {"output": response})
    result = qa({"question": query})
    return result["answer"]

# custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in German language.' If you do not know the answer reply with 'I am sorry'.
#         Chat History:
#         {chat_history}
#         Follow Up Input: {question}
#         Standalone question:"""
#         CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)


# def startt(base_url):
#
#     if base_url[-1] == '/':
#         base_url = base_url[:-1]
#
#     # Scrape the main url
#     docs_transformed = scraper(base_url)
#
#     full_data.append(docs_transformed)
#
#     # Find nav links of baseURL
#     paths = find_navlinks(docs_transformed)
#
#     # Scrape data of the nav links
#     for path in paths:
#         docs_transformed = scraper(base_url+path)
#         x = check_new_navlinks(docs_transformed, paths)
#         if x is not None:
#             paths.append(x)
#             full_data.append(docs_transformed[0].page_content)
#     print(paths)
#
#     # After scraping everywhere we collect a list of links and remove some links such as twitter, insta etc.
#     new_links = list(set(links))
#     new_links = [link for link in new_links if not any(domain in link for domain in exclude_domains)]
#     try:
#         new_links.remove(base_url+'/')
#         new_links.remove(base_url)
#     except Exception as e:
#         print()
#     print(new_links)
#
#     # Scrape data from those links also
#     if new_links is not None:
#         for path in new_links:
#             docs_transformed = scraper(path)
#             full_data.append(docs_transformed[0].page_content)
#
#     return True
