import argparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_name", type=str, default="distilbert-base-multilingual-cased")
    args = parse.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Model and tokenizer saved to HuggingFace cache (typically at ~/.cache/huggingface)")
