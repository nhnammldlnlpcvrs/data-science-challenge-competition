from transformers import AutoTokenizer

# Load tokenizer CafeBERT
tokenizer = AutoTokenizer.from_pretrained("cafebiz/cafebert-base")

def preprocess(example, max_len_context=128, max_len_prompt=64, max_len_response=128):
    """
    Biến đổi một sample {context, prompt, response, label} 
    thành input mà Multi-Input CafeBERT có thể xử lý.

    Args:
        example (dict): 1 dòng dữ liệu từ dataset
        max_len_context (int): max length cho context
        max_len_prompt (int): max length cho prompt
        max_len_response (int): max length cho response

    Returns:
        dict: chứa input_ids và attention_mask cho từng feature + label
    """

    # Encode riêng từng feature
    context_enc = tokenizer(
        example["context"], truncation=True, padding="max_length", max_length=max_len_context
    )
    prompt_enc = tokenizer(
        example["prompt"], truncation=True, padding="max_length", max_length=max_len_prompt
    )
    response_enc = tokenizer(
        example["response"], truncation=True, padding="max_length", max_length=max_len_response
    )

    return {
        "context_input_ids": context_enc["input_ids"],
        "context_attention_mask": context_enc["attention_mask"],
        "prompt_input_ids": prompt_enc["input_ids"],
        "prompt_attention_mask": prompt_enc["attention_mask"],
        "response_input_ids": response_enc["input_ids"],
        "response_attention_mask": response_enc["attention_mask"],
        "labels": example["label"],
    }