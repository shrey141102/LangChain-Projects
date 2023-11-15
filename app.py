from utils import  memory_query, starter_code, choose_llm, choose_embeddings
import os
import streamlit as st

os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # or 'false' as needed

## run app using `streamlit run app.py`
def main():
    st.set_page_config(
        page_title="ChatBot", page_icon=":cat:")

    st.title("AI Chatbot 🤖")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.sidebar.title("Please choose your options here before proceeding to the chat section.")
    llm_choose = st.sidebar.selectbox('Choose your LLM?',("ChatGooglePalm", "ChatOpenAI", "HuggingFaceHub"))
    embed_choose = st.sidebar.selectbox('Choose your Embeddings?',("HuggingFaceEmbeddings", "OpenAIEmbeddings", "SentenceTransformerEmbeddings"))

    OPENAI_API = st.sidebar.text_input('Enter API key for OpenAI')
    HUGGINGFACE_API = st.sidebar.text_input('Enter API key for Huggingface')
    GOOGLE_PALM_API_KEY = st.sidebar.text_input('Enter API key for Google Palm API')

    if HUGGINGFACE_API is not None:
        variable_name = "HUGGINGFACEHUB_API_TOKEN"
        variable_value = HUGGINGFACE_API
        os.environ[variable_name] = variable_value

    if GOOGLE_PALM_API_KEY is not None:
        variable_name = "GOOGLE_API_KEY"
        variable_value = GOOGLE_PALM_API_KEY
        os.environ[variable_name] = variable_value

    if OPENAI_API is not None:
        variable_name = "OPENAI_API_KEY"
        variable_value = OPENAI_API
        os.environ[variable_name] = variable_value

    base_url = st.sidebar.text_input("Enter the website url?", value="https://")
    button = st.sidebar.button("All details filled! ✅")

    if base_url and button:
        with st.spinner("Scraping the site..."):
            if starter_code(base_url):
                st.sidebar.success('Scraping completed!🥳🥳')

    if base_url and llm_choose and embed_choose is not None:
        if prompt := st.chat_input("What is up?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = memory_query(query=prompt,
                                    llm=choose_llm(llm_choose),
                                    embeddings=choose_embeddings(embed_choose))
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.header("Please fill in the values of sidebar to proceed further ⚠️")

if __name__ == "__main__":
    main()

