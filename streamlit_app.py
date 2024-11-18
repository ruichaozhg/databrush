import streamlit as st
from typing import Generator
from groq import Groq
import streamlit.components.v1 as components
import logging
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import streamlit_analytics
from PIL import Image
from google.oauth2 import service_account
import gspread
from datetime import datetime
logging.info("This is a logging test")
logging.basicConfig(level=logging.INFO)


def append_to_sheet(prompt, generated, answer):
    """
    Add to GSheet
    """

    service_json = {}
    service_json["type"] = st.secrets["type"]
    service_json["project_id"] = st.secrets["project_id"]
    service_json["private_key_id"] = st.secrets["private_key_id"]
    service_json["private_key"] = st.secrets["private_key"]
    service_json["client_email"] = st.secrets["client_email"]
    service_json["client_id"] = st.secrets["client_id"]
    service_json["auth_uri"] = st.secrets["auth_uri"]
    service_json["token_uri"] = st.secrets["token_uri"]
    service_json["auth_provider_x509_cert_url"] = st.secrets["auth_provider_x509_cert_url"]
    service_json["client_x509_cert_url"] = st.secrets["client_x509_cert_url"]
    service_json["universe_domain"] = st.secrets["universe_domain"]
    credentials = service_account.Credentials.from_service_account_info(
        service_json,
        scopes=["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/drive']
    )
    gc = gspread.authorize(credentials)
    sh = gc.open_by_url(st.secrets["PRIVATE_GSHEETS_URL"])
    worksheet = sh.get_worksheet(0) # Assuming you want to write to the first sheet
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    worksheet.append_row([current_time,prompt, generated, answer])







streamlit_analytics.start_tracking()

st.set_page_config(page_icon="🧼", layout="centered",
                page_title="Privacy Search")
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
    st.image("images/11.png")
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''

    st.markdown(hide_img_fs, unsafe_allow_html=True)
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


if prompt := st.chat_input("Search or ask a compliance question..."):

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
        append_to_sheet(prompt, True, full_response)

    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
        append_to_sheet(prompt, True, combined_response)

# save_to_json="/workspaces/groq_streamlit_demo/file.json"
streamlit_analytics.stop_tracking()
