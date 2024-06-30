import streamlit as st


class MultiPage:
    def __init__(self):
        self.pages = []

    def add_page(self, title, func):
        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self):
        if 'pytorch_checked' not in st.session_state:
            st.session_state.pytorch_checked = False

        if not st.session_state.pytorch_checked:
            # Directly run the first page if PyTorch environment has not been checked
            self.pages[0]['function'](self.pages)
        else:
            page = st.sidebar.selectbox(
                'App Navigation',
                self.pages,
                format_func=lambda page: page['title'],
                key='page'
            )
            page['function'](self.pages)
