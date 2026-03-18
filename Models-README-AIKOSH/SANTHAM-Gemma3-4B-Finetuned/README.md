# SANTHAM-Gemma3-4B-Finetuned

## About the Model
SANTHAM-Gemma3-4B-Finetuned is a Sanskrit → Tamil translation model built on the Gemma 3 (4B) architecture.  
It is trained on a parallel corpus developed as part of the Sanskrit Knowledge Accessor project, enabling it to capture linguistic nuances and generate fluent Tamil translations from classical Sanskrit inputs.

---

## Setup Instructions

### 1. Install uv
This project uses uv for fast and reproducible dependency management.

**macOS/Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

### 2. Project Setup
Initialize the environment and install dependencies:

```bash
uv sync
```

---

### 3. Authentication (Hugging Face)

You need a Hugging Face access token to download Gemma weights.

Steps:
1. Generate a Read Token from Hugging Face  
2. Accept license terms for the Gemma 3 4B model  
3. Create a `.env` file:

```env
HUGGINGFACE_TOKEN=hf_your_token_here
```

---

## Model Inference Code

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables and login
load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

MODEL_NAME = "google/gemma-3-4b-it"
LORA_PATH = "SANTHAM-Gemma3-4B-FT"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model on {DEVICE}...")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto" if DEVICE == "cuda" else None,
)

# Merge LoRA weights
print("Merging LoRA weights...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.merge_and_unload()
model.eval()


def run_translation(texts, src_code="san_Deva", tgt_code="tam_Taml", batch_size=8):
    """
    Translate Sanskrit text to Tamil.
    Supports both single string and batch inputs.
    """

    lang_map = {
        "san_Deva": "Sanskrit",
        "tam_Taml": "Tamil"
    }

    src_lang = lang_map[src_code]
    tgt_lang = lang_map[tgt_code]

    # Normalize input
    is_single = isinstance(texts, str)
    input_list = [texts] if is_single else texts

    results = []

    for i in tqdm(range(0, len(input_list), batch_size), desc="Processing"):
        batch = input_list[i : i + batch_size]

        # Build prompts
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": f"Translate the following text from {src_lang} to {tgt_lang}:\n\n{text}"
                    }
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for text in batch
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                repetition_penalty=1.1,
                do_sample=False
            )

        # Decode generated tokens only
        input_len = inputs.input_ids.shape[1]

        for output_ids in outputs:
            decoded = tokenizer.decode(
                output_ids[input_len:],
                skip_special_tokens=True
            ).strip()
            results.append(decoded)

    return results[0] if is_single else results


# Example: Single input
single_input = "नमो नमः"
print(f"\nSingle Result: {run_translation(single_input)}")

# Example: Batch input
batch_input = ["अहं गच्छामि", "त्वं कुत्र गच्छसि?"]
print(f"\nBatch Results: {run_translation(batch_input, batch_size=2)}")
```

---

## Run Inference

```bash
uv run python inference.py
```

---

## Note
If you encounter a CUDA Out Of Memory error, reduce the batch size:

```python
batch_size = 4
```
