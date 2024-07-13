import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


def summarize_text(text):
    model_name = 'tuner007/pegasus_summarizer'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    batch = tokenizer([text], truncation=True, padding='longest', max_length=1024, return_tensors="pt").to(torch_device)
    gen_out = model.generate(**batch, max_length=128, num_beams=5, num_return_sequences=1, temperature=1.5)
    output_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0]  # Convert list to string
    return output_text.replace(" ", "")  # Remove spaces between letters
