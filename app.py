import os
import threading

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from transformers.generation.streamers import TextIteratorStreamer

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_id = "microsoft/bitnet-b1.58-2B-4T"


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)


def generate_response(message: str, history):
    messages = history

    # 現在のユーザーメッセージを追加
    messages.append({"role": "user", "content": message})

    # チャットテンプレートを適用
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # トークナイズ
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # ストリーマーを設定
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # 生成パラメータ
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=500,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # 別スレッドで実行
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 生成された応答を少しずつyield
    partial_message = ""
    for token in streamer:
        partial_message += token
        yield partial_message


# Gradio ChatInterfaceの設定
def main():
    gr.ChatInterface(
        fn=generate_response,
        title="BitNet Chat Assistant",
        description="BitNet-b1.58-2B-4Tモデルを使用したチャットアシスタントです。",
        examples=[
            "こんにちは、元気ですか？",
            "AIについて教えてください",
            "プログラミングの良い勉強方法は？",
        ],
        theme="soft",
        type="messages",  # 追加: OpenAIスタイルのメッセージ形式を使用
    ).launch(share=False)


if __name__ == "__main__":
    main()
