import torch
from transformers import pipeline
import pandas as pd
import csv

out = csv.writer(open('model_run_output.csv', 'w'))
out.writerow(['intruction', 'context', 'model', 'output'])
model_name = "databricks/dolly-v2-12b"
generate_text = pipeline(model=model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")



data = pd.read_json("eval_absa_new.json")
for idx,k in data.iterrows():
    instruct = k["instruction"]
    context = k["context"]
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n### Input:\n{}\n\n### Response:""".format(instruct,context)
    out.writerow([instruct,context,model_name,generate_text(prompt)[0]['generated_text']])

# Dependencies
# pip install "accelerate>=0.16.0,<1" "transformers[torch]>=4.28.1,<5" "torch>=1.13.1,<2"
