import streamlit as st
from typing import Generator
from groq import Groq
import streamlit.components.v1 as components
import logging
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import streamlit_analytics
from PIL import Image


logging.info("This is a logging test")
logging.basicConfig(level=logging.INFO)



streamlit_analytics.start_tracking()

st.set_page_config(page_icon="🧼", layout="centered",
                page_title="FMHY Search")
# components.html(
# """
# <!-- Google tag (gtag.js) -->
# <script async src="https://www.googletagmanager.com/gtag/js?id=G-HQ0Y0CEC14"></script>
# <script>
#   window.dataLayer = window.dataLayer || [];
#   function gtag(){dataLayer.push(arguments);}
#   gtag('js', new Date());

#   gtag('config', 'G-HQ0Y0CEC14');
# </script>
# """
# )


col1, col2, col3 = st.columns((1, 4, 1))
with col2:
    st.image(Image.open("images/11.png"))
# st.logo("images/11.png",link="https://protrustai.com")

# st.subheader("FMHY Search", divider="rainbow", anchor=False)
# groq_api_key = st.text_input("Paste your groq key here:")

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3-70b-8192"

# Define model details
models = {
    "llama3-groq-70b-8192-tool-use-preview":{"name": "llama3-groq-70b-8192-tool-use-preview", "tokens": 8192, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Layout for model selection and max_tokens slider
# col1 = st.columns(1)
# with st.container():
with st.container():

    model_option = "llama3-70b-8192"
# Detect model change and clear chat history if model has changed
# if st.session_state.selected_model != model_option:
#     st.session_state.messages = []
#     st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

# with col2:
#     # Adjust max_tokens slider dynamically based on the selected model
#     max_tokens = st.slider(
#         "Max Tokens:",
#         min_value=512,  # Minimum value to allow some flexibility
#         max_value=max_tokens_range,
#         # Default value or max allowed if less
#         value=min(32768, max_tokens_range),
#         step=512,
#         help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
#     )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🧽️' if message["role"] == "assistant" else '😃'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if prompt := st.chat_input("Search..."):

    if prompt == "import demo google sheet":
        url = "https://docs.google.com/spreadsheets/d/1jxXhwp7O1Tc-ZETzFZBs3_YqhjT7C0veJ-7dk_1MRCA/edit?usp=sharing"

        conn = st.connection("gsheets", type=GSheetsConnection)

        data = conn.read(spreadsheet=url)
        st.markdown(prompt)
        
        # change prompt value to the dataset
        prompt = "\n".join(str(item) for item in [data])

        st.session_state.messages.append({"role": "user", "content": prompt})

    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='😃'):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens_range,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🧽️"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
# save_to_json="/workspaces/groq_streamlit_demo/file.json"
streamlit_analytics.stop_tracking()
