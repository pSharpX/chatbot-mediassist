import os
import threading
from dataclasses import dataclass
from typing import List, Dict, Any
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, GenerationConfig

@dataclass
class GenParams:
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
    repetition_penalty: float = 1.05
    do_sample: bool = True

def get_device_dtype():
    """Choose a device & dtype that works widely."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    # Apple Silicon (M1/M2)
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_name: str):
    device, dtype = get_device_dtype()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Try to load efficiently; fall back gracefully.
    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    # bitsandbytes 4-bit (optional)
    use_bnb = os.environ.get("USE_BNB", "0") == "1"
    if use_bnb:
        kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": dtype,
        })

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception:
        # Fallback: load on CPU full precision if needed
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()
    # Move to device if not already handled by bnb
    if not use_bnb and device != "cpu":
        model.to(device)

    return model, tok, device

def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Use the tokenizer's chat template when available, otherwise a simple fallback."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Simple generic prompt style
        sys_msgs = "\n".join([m["content"] for m in messages if m["role"] == "system"]) or "You are a helpful assistant."
        convo = "".join(
            f"\nUser: {m['content']}\nAssistant:" if m["role"] == "user" else f" {m['content']}\n"
            for m in messages if m["role"] != "system"
        )
        return sys_msgs + "\n" + convo

def stream_generate(model, tokenizer, device, prompt_text: str, gen: GenParams):
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    if device != "cpu":
        input_ids = input_ids.to(device)

    gen_config = GenerationConfig(
        max_new_tokens=gen.max_new_tokens,
        temperature=gen.temperature,
        top_p=gen.top_p,
        do_sample=gen.do_sample,
        repetition_penalty=gen.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    kwargs = dict(
        input_ids=input_ids,
        generation_config=gen_config,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    for new_text in streamer:
        yield new_text