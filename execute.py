from openai import OpenAI
from LLM import run_llm

config = {
    'data': 'sst-5',
    'method': 'LENS',
    'client': OpenAI(api_key='your-api-key', base_url='your-url'),
    'model': 'deepseek-chat',
    'batch_size': 30,
    'valid_size': 100,
    'data_size': 2000,
    'progressive_factor': 2,
    'desired_candidate_size': 100,
    'initial_score_size': 20,
    'iter_num': 5,
    'beam_size': 6,
    'substitution_size': 2,
    'alpha': 1
}

run_llm(config)