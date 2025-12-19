import json
import time
import torch
from transformers import AutoModelForSequenceClassification
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def hallu_vectara(model):
    # Setup
    input_path = "tree_number_test_with_summaries_"+ model + ".txt"
    output_path = "tree_number_test_with_summaries_"+model+"_with_herman_hallucinations.txt"
    batch_size = 1  # Adjust depending on your RAM/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        'vectara/hallucination_evaluation_model', trust_remote_code=True
    ).to(device)
    
    start_time = time.time()
    
    def get_batches(file_obj, batch_size):
        """Yield batches of parsed JSON lines"""
        batch = []
        for line in file_obj:
            if line.strip():
                item = json.loads(line)
                article = item.get("article_text", "")
                summary = item.get("summary", "")
                batch.append((item, (article, summary)))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
    
    processed = 0
    
    with open(input_path, 'r', encoding='utf-8') as in_f, open(output_path, 'w', encoding='utf-8') as out_f:
        for batch in get_batches(in_f, batch_size):
            items, pairs = zip(*batch)
            scores = model.predict(list(pairs))  # Predict on a batch
    
            # Convert to list if needed
            if torch.is_tensor(scores):
                scores = scores.tolist()
    
            for item, score in zip(items, scores):
                item["hallucination_vectara"] = float(score)
                out_f.write(json.dumps(item) + "\n")
                processed += 1
    
            # 🔽 Clear memory after batch
            del scores, items, pairs
            torch.cuda.empty_cache()
            
    end_time = time.time()
    print(f"✅ Done. Processed {processed} items.")
    print(f"🕒 Inference took {end_time - start_time:.2f} seconds.")

hallu_vectara("Meditron3-Qwen2.5-14B-adapter")
hallu_vectara("Meditron3-Qwen2.5-7B-adapter")
