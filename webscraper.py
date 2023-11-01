from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
import os
import re
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.chroma import Chroma

#Make a file apikey.py and add your api key there, and import like this
from apikey import OPENAI_API

variable_name = "OPENAI_API_KEY"
variable_value = OPENAI_API
os.environ[variable_name] = variable_value

link_pattern = r'(https?://\S+)'
links = []
file_name = "data.txt"
base_url = input("enter the url: ")

def main():

    docs_transformed = scraper(base_url)

    with open(file_name, "w") as file:
        file.write(f'This is the data from {base_url}\n')
        file.write(docs_transformed[0].page_content)

    paths = find_navlinks(docs_transformed)

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

    new_links = list(set(links))
    print(new_links)

    for path in new_links:
        docs_transformed = scraper(path)
        # x = check_new_navlinks(docs_transformed, paths)
        # if x is not None:
        #     paths.append(x)
        with open(file_name, "a") as file:
            file.write("\n")
            file.write("\n")
            file.write(f'This is the data from {path}\n')
            file.write(docs_transformed[0].page_content)

    memory_query()

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
    html = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p", "li", "div", "a", "span"])

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

def normal_query():
    loader = TextLoader(file_name)
    index = VectorstoreIndexCreator().from_loaders([loader])
    query = input("What do you want to ask?: ")
    print(index.query(query))

    while True:
        query = input("Any other questions?: ")
        if query == "no":
            break
        print(index.query(query))

def memory_query():
    loader = TextLoader(file_name)
    documents = loader.load()

    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=80)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)

    while True:
        query = input("Enter your query?: ")
        result = qa({"question": query})
        print(result["answer"])

if __name__ == "__main__":
    main()
