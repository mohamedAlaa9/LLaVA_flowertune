from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration
class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["image"])

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

# def formatting_prompts_func(example):
#     output_texts = []
#     print(example[:10000])
#     # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
#     mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
#     for i in range(len(example["content"])):
#         text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
#         output_texts.append(text)
#     return output_texts
# def formatting_prompts_func(example):
#     output_texts = []
#     # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
#     mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
#     for i in range(0, len(example["content"]["text"]), 2):
#         text = f"{mssg}\n### Instruction:\n{example['content']['text'][i]}\n### Response: {example['content']['text'][i+1]}"
#         output_texts.append(text)
#     return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    # From: https://huggingface.co/docs/trl/en/sft_trainer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name, use_fast=True, padding_side="right"
    # )
    # tokenizer.pad_token = tokenizer.eos_token



    tokenizer = AutoTokenizer.from_pretrained(model_name)
    LLAVA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}"""
    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer = tokenizer
    # response_template_with_context = "\n### Response:"  # alpaca response tag
    # response_template_ids = tokenizer.encode(
    #     response_template_with_context, add_special_tokens=False
    # )[2:]
    # data_collator = DataCollatorForCompletionOnlyLM(
    #     response_template_ids, tokenizer=tokenizer
    # )
    data_collator = LLavaDataCollator(processor)
    return tokenizer, data_collator