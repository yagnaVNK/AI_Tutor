import streamlit as st
import ollama

def call_llm(prompt, model="llama3:8b", chat_history=[]):
    try:
        system_prompt = "You are a funny helpful ai assistant who gives the answers in a short and conversational way."
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": role, "content": content} for role, content in chat_history])
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=model,
            messages=messages
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

def get_available_models():
    try:
        models = ollama.list_models()
        return [model['name'] for model in models]
    except Exception as e:
        return [f"Error fetching models: {e}"]

def main():
    st.set_page_config(page_title="LLM Chatbot using Ollama", layout="centered")
    st.title("🤖 Chatbot using Ollama LLaMA")

    st.sidebar.title("Settings")
    models = ["llama3:8b","llama3.2:1b"]
    model = st.sidebar.selectbox("Choose LLaMA Model", models)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        role, content = chat
        if role == "user":
            st.chat_message("user").markdown(content)
        else:
            col1, col2 = st.columns([0.9, 0.1])
            
            with col1:
                st.chat_message("assistant").markdown(content)
            with col2:
                 if st.button("📋", key=f"copy_{len(st.session_state.chat_history)}"):
                    st.session_state.clipboard_text = content
                    st.success("Copied to clipboard!")

    prompt = st.chat_input("Ask something...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append(("user", prompt))

        response = call_llm(prompt, model, st.session_state.chat_history)
        st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append(("assistant", response))

if __name__ == "__main__":
    main()
