import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from peft import PeftModel


def create_summary(base_model_path, adapter_path, output_suffix):
    # ⚠️ Better to use an env variable in real code, but keeping your pattern here
    # login("here insert your key")

    # 🔹 Load tokenizer from the BASE model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # 🔹 Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        device_map="cuda",
    )

    # 🔹 Apply adapter on top of base model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to("cuda")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token

    # Load input data
    source_path = "tree_number_test.txt"
    with open(source_path, "r") as file:
        data = [json.loads(line) for line in file]

    start_time = time.time()

    for i, entry in enumerate(data):
        input_text = entry.get("article_text", "")
        prompt = (
            "Summarize the following biomedical article in a clear and concise manner, "
            "in no more than 300 words:\n\n"
            f"{input_text}\n\nSummary:"
        )

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to("cuda")

        # Generate
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
            )

        # Decode and extract only the summary part
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        summary = full_output.split("Summary:")[-1].strip()

        entry["summary"] = summary

        # Free up memory
        del inputs, output
        torch.cuda.empty_cache()
        print(f"[{i+1}/{len(data)}] Summary complete")

    elapsed_time = time.time() - start_time
    print(f"\n✅ All summaries completed in {elapsed_time:.2f} seconds.")

    # Save output
    output_path = f"tree_number_test_with_summaries_{output_suffix}.txt"
    with open(output_path, "w") as outfile:
        for entry in data:
            outfile.write(json.dumps(entry) + "\n")

    print(f"✅ Updated file saved to: {output_path}")


create_summary(
    base_model_path="./Meditron3-Qwen2.5-7B",
    adapter_path="./Meditron3-Qwen2.5-7B-text-generation",
    output_suffix="Meditron3-Qwen2.5-7B-adapter"
)

create_summary(
    base_model_path="./Meditron3-Qwen2.5-14B",
    adapter_path="./Meditron3-Qwen2.5-14B-text-generation",
    output_suffix="Meditron3-Qwen2.5-14B-adapter"
)