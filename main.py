import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/rut5_base_headline_gen_telegram")
model = AutoModelForSeq2SeqLM.from_pretrained("IlyaGusev/rut5_base_headline_gen_telegram")

article_text = "Первую многоножку, у которой более тысячи ногтам окру"
def modelFoo(text):
    input_ids = tokenizer(
        [text],
     max_length=100,
     add_special_tokens=True,
     padding="max_length",
     truncation=True,
     return_tensors="pt"
    )["input_ids"]

    output_ids = model.generate(input_ids=input_ids)[0]

    headline = tokenizer.decode(output_ids, skip_special_tokens=True)
    return (headline)

def main():
    st.write(modelFoo(article_text))

if __name__ == "__main__":
    main()
