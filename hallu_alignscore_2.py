import json
import os
import torch
import sys
from transformers import AutoTokenizer
from tqdm import tqdm  # optional, for progress

torch.cuda.empty_cache()

# Set up AlignScore import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from alignscore import AlignScore

# Set device and paths
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def hallu_alignscore(model):
    input_path = f"tree_number_test_with_summaries_{model}_with_minicheck_hallucinations.txt"
    output_path = f"tree_number_test_with_summaries_{model}_with_alignscore_hallucinations.txt"

    # Initialize tokenizer and AlignScore model
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    scorer = AlignScore(
        model='roberta-large',
        batch_size=1,
        device='cuda:0',
        ckpt_path='/home/kristo/pycharm/AlignScore-main/AlignScore-large.ckpt',
        evaluation_mode='nli_sp'
    )

    # Load data
    items = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            article = item.get("article_text", "")
            summary = item.get("summary", "")

            if isinstance(article, list):
                article = "\n".join(article)
            if isinstance(summary, list):
                summary = " ".join(summary)

            item["article_text"] = article
            item["summary"] = summary
            items.append(item)

    # Run AlignScore in batches
    batch_size = 8
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(items), batch_size), desc=f"Scoring {model}"):
            batch = items[i:i+batch_size]
            batch_docs = []
            batch_claims = []
            valid_indices = []

            # Pre-check for empty inputs
            for idx, item in enumerate(batch):
                article = item["article_text"]
                summary = item["summary"]

                # Tokenize and decode truncated versions
                max_tokens_each = 1000
                article_ids = tokenizer(article, truncation=False, max_length=max_tokens_each, return_tensors="pt")["input_ids"][0]
                summary_ids = tokenizer(summary, truncation=False, max_length=max_tokens_each, return_tensors="pt")["input_ids"][0]

                article_decoded = tokenizer.decode(article_ids, skip_special_tokens=True).strip()
                summary_decoded = tokenizer.decode(summary_ids, skip_special_tokens=True).strip()

                if not article_decoded or not summary_decoded:
                    item["hallucination_alignscore_roberta"] = -1.0
                    f_out.write(json.dumps(item) + "\n")
                else:
                    batch_docs.append(article_decoded)
                    batch_claims.append(summary_decoded)
                    valid_indices.append(idx)

            # Run AlignScore if valid inputs exist
            if batch_docs:
                try:
                    scores = scorer.score(contexts=batch_docs, claims=batch_claims)
                except torch.cuda.OutOfMemoryError:
                    print(f"[⚠️ OOM] Skipping items {i}-{i+batch_size-1}")
                    torch.cuda.empty_cache()
                    continue

                for j, score in enumerate(scores):
                    item = batch[valid_indices[j]]
                    item["hallucination_alignscore_roberta"] = float(score)
                    f_out.write(json.dumps(item) + "\n")

            torch.cuda.empty_cache()

    print(f"✅ Done. Processed {len(items)} items for model: {model}")



# hallu_alignscore(file_path,"biomistral7b")
hallu_alignscore("Meditron3-Qwen2.5-7B-adapter")
hallu_alignscore("Meditron3-Qwen2.5-14B-adapter")
