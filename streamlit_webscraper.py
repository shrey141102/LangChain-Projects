import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import os
import re
os.system("playwright install")

st.title("Web Scraper and Text Search App")

# Get OpenAI API Key from the user
variable_name = "OPENAI_API_KEY"
variable_value = st.text_input("Enter OpenAI API Key:")
os.environ[variable_name] = variable_value

file_name = "data.txt"
base_url = st.text_input("Enter the URL:")

def scraper(url):
    loader = AsyncChromiumLoader([url])
    html = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p", "li", "div", "a", "span"])

    return docs_transformed

def find_navlinks(docs_transformed):
    paths = re.findall(r'\(/\w+\)', docs_transformed[0].page_content)
    filtered_paths = [path[1:-1] for path in paths]
    unique_paths = list(set(filtered_paths))
    return unique_paths

if st.button("Scrape Data"):
    st.write(f"Scraping data from {base_url}...")

    docs_transformed = scraper(base_url)

    with open(file_name, "w") as file:
        file.write(f'This is the data from {base_url}\n')
        file.write(docs_transformed[0].page_content)

    paths = find_navlinks(docs_transformed)

    for path in paths:
        docs_transformed = scraper(base_url + path)
        with open(file_name, "a") as file:
            file.write("\n")
            file.write("\n")
            file.write(f'This is the data from {base_url + path}\n')
            file.write(docs_transformed[0].page_content)

    st.write(f"Scraped data from {len(paths)} additional pages.")
    st.write("Data saved to data.txt")

# Text search
query = st.text_input("Enter your query:")
if st.button("Search"):
    index = VectorstoreIndexCreator().from_loaders([TextLoader(file_name)])
    results = index.query(query, verbose = True)
    st.write("Search Results:")
    st.write(results)


