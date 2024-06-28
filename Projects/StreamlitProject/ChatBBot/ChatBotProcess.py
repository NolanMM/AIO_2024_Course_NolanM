import streamlit as st
from hugchat import hugchat
from hugchat.login import Login


class ChatBotApp:
    def __init__(self):
        self.hf_email = ""
        self.hf_pass = ""

    def login_section(self):
        with st.sidebar:
            st.title('Login HugChat')
            self.hf_email = st.text_input('Enter E-mail:')
            self.hf_pass = st.text_input('Enter Password:', type='password')
            if not (self.hf_email and self.hf_pass):
                st.warning('Please enter your account!')
            else:
                st.success('Proceed to entering your prompt message!')

    def initialize_session(self):
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "How may I help you?"}]

    def display_messages(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    def generate_response(self, prompt_input, email, passwd):
        sign = Login(email, passwd)
        cookies = sign.login()
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        return chatbot.chat(prompt_input)

    def handle_user_input(self):
        if prompt := st.chat_input(disabled=not (self.hf_email and self.hf_pass)):
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = self.generate_response(
                            prompt, self.hf_email, self.hf_pass)
                        st.write(response)
                        message = {"role": "assistant", "content": response}
                        st.session_state.messages.append(message)

    def run(self):
        st.title('Simple ChatBot')
        self.login_section()
        self.initialize_session()
        self.display_messages()
        self.handle_user_input()
