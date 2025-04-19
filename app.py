import asyncio
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from transformers.generation.streamers import TextStreamer

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_id = "microsoft/bitnet-b1.58-2B-4T"


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# Apply the chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "How are you?"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)


class AsyncStreamer(TextStreamer):
    def __init__(self, tokenizer, queue, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.queue = queue

    def on_text(self, text, **kwargs):
        asyncio.create_task(self.queue.put(text))

    def on_end(self):
        asyncio.create_task(self.queue.put(None))


async def completion():
    queue = asyncio.Queue()
    streamer = AsyncStreamer(
        tokenizer,
        queue,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    # model.generateをバックグラウンドで実行
    loop = asyncio.get_event_loop()

    def run_generate():
        model.generate(
            **chat_input,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    loop.run_in_executor(None, run_generate)
    while True:
        token = await queue.get()
        if token is None:
            break
        yield token


async def main():
    # 非同期でメッセージを受信
    async for message in completion():
        print(message, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
