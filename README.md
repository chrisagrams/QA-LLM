# PubMedQA Evaluation Script
This script evaluates the performance of a ChatGPT-compatible API (OpenAI, vLLM, LM Studio, etc.) on the [PubMedQA](https://pubmedqa.github.io/) dataset. It submits PubMedQA questions to the API and compares the generated answers against the ground truth.

## Usage
Run the script with:
``` sh
python script.py --url <API_URL> --api_key <API_KEY> --model <MODEL_NAME> --test_set <TEST_SET_PATH> --output_prefix <OUTPUT_PREFIX>
```

### Arguments
| Argument | Description | Default |
| -------- | ----------- | ------- |
| `--url`  | Base URL of the API | `http://localhost:1234/v1` |
| `--api_key` | API key for authentication | `token-abc123` |
| `--model` | Model name to use for the API requst | *Required* | 
| `--test_set` | Path to the PubMedQA test set JSON file | `data/pubmedqa/test_set.json` |
| `--output_prefix` | Prefix for the output CSV file | `""` | 

## Output
The script generates a CSV file named:

``` sh
<MODEL_NAME><OUTPUT_PREFIX>-pubmedqa_results.csv
```

Each row in the CSV contains:
| Column | Description |
| ------ | ----------- | 
| key | Question identifier from dataset |
| answer | Model's predicted answer (`yes`, `no`, `maybe`) |
| truth | Ground truth answer from dataset |

### Example Run
```sh
python main.py --model=deepseek-r1-distill-qwen-7b --output_prefix=-test
```
Output:
```sh
Key: 12377809 Answer: yes Truth: yes Prompt Tokens: 358 Completion Tokens: 347 Total Tokens: 705
Key: 26163474 Answer: yes Truth: yes Prompt Tokens: 467 Completion Tokens: 215 Total Tokens: 682
Key: 19100463 Answer: yes Truth: yes Prompt Tokens: 378 Completion Tokens: 391 Total Tokens: 769
...
Results saved to deepseek-r1-distill-qwen-7b-test-pubmedqa_results.csv
```