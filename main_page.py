import streamlit as st
import helpers

# ------------------------------
# Defaults & Utilities
# ------------------------------
DEFAULT_MODELS = {
    "TinyLlama 1.1B Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Gemma 2B It": "google/gemma-2b-it",
    "Zephyr 7B Beta": "HuggingFaceH4/zephyr-7b-beta",
    "Mistral 7B Instruct v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
}


# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("⚙️ Settings")
model_label = st.sidebar.selectbox("Model", list(DEFAULT_MODELS.keys()), index=0)
model_name = DEFAULT_MODELS[model_label]

col1, col2 = st.sidebar.columns(2)
with col1:
    temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
with col2:
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

max_new_tokens = st.sidebar.slider("Max new tokens", 16, 2048, 512, 16)
repetition_penalty = st.sidebar.slider("Repetition penalty", 1.0, 2.0, 1.05, 0.01)

system_prompt = st.sidebar.text_area(
    "System prompt",
    value=(
        "You are a helpful, concise assistant. If you use code, explain briefly."
    ),
    height=100,
)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear chat"):
    st.session_state["messages"] = [{"role": "system", "content": system_prompt}]
    st.rerun()


# ------------------------------
# Session State
# ------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt}
    ]

# Update system message if changed
if st.session_state["messages"] and st.session_state["messages"][0]["role"] == "system":
    st.session_state["messages"][0]["content"] = system_prompt


# ------------------------------
# Main App
# ------------------------------
st.title("🤖 Open-Source LLM Chatbot")
st.caption("Built with Streamlit + Transformers (Hugging Face)")

# Load model/tokenizer
with st.spinner(f"Loading model: {model_label} ..."):
    model, tokenizer, device = helpers.load_model_and_tokenizer(model_name)

# Show history
for m in st.session_state["messages"]:
    if m["role"] == "system":
        continue
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

# Chat input
user_msg = st.chat_input("Type your message…")
if user_msg:
    st.session_state["messages"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Prepare prompt
    prompt_text = helpers.apply_chat_template(tokenizer, st.session_state["messages"])

    # Generate & stream
    with st.chat_message("assistant"):
        gen = helpers.GenParams(
            temperature=temp,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        placeholder = st.empty()
        buf = ""
        for chunk in helpers.stream_generate(model, tokenizer, device, prompt_text, gen):
            buf += chunk
            placeholder.markdown(buf)
        st.session_state["messages"].append({"role": "assistant", "content": buf})


# ------------------------------
# Footer
# ------------------------------
st.markdown(
    "<hr />\n<p style='font-size: 0.9rem;'>💡 Tip: For best results on CPU, try smaller models like TinyLlama. For higher quality, use a GPU and models like Zephyr or Mistral. Set USE_BNB=1 to enable 4-bit loading if bitsandbytes is installed.</p>",
    unsafe_allow_html=True,
)