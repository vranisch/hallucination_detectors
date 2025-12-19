import json
import os
from minicheck.minicheck import MiniCheck

# Set GPU (or CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def hallu_minicheck(model):
    # File paths
    input_path = "tree_number_test_with_summaries_"+ model +"_with_herman_hallucinations.txt"
    output_path = "tree_number_test_with_summaries_"+ model +"_with_minicheck_hallucinations.txt"



    # Load MiniCheck model
    scorer = MiniCheck(model_name='roberta-large', cache_dir='./ckpts')

    # Load all examples
    docs, claims, items = [], [], []

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

            docs.append(article)
            claims.append(summary)
            items.append(item)

    # Run inference
    pred_labels, raw_probs, _, _ = scorer.score(docs=docs, claims=claims)

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item, prob in zip(items, raw_probs):
            item["hallucination_minicheck"] = float(prob)
            f_out.write(json.dumps(item) + "\n")

    print(f"✅ Done. Processed {len(items)} items.")



hallu_minicheck("Meditron3-Qwen2.5-14B-adapter")
hallu_minicheck("Meditron3-Qwen2.5-7B-adapter")