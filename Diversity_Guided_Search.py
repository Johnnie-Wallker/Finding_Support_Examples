import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from Progressive_Example_Filtering import get_filtered_data


def sample_calculator(examples, permutation, config):
    s = []
    for i in range(len(examples)):
        example = examples.iloc[i, :]
        similarities = []
        for j in range(len(permutation)):
            sample = permutation.iloc[j, :]
            columns_to_drop = ['comment', 'label', 'info']
            f_sample = sample.drop(columns_to_drop).values.tolist()
            f_example = example.drop(columns_to_drop).values.tolist()
            f_sample = np.array(f_sample).reshape(1, -1)
            f_example = np.array(f_example).reshape(1, -1)
            similarities.append(cosine_similarity(f_sample, f_example))
        s.append(example['info'] - config['alpha'] * sum(similarities))
    index = s.index(max(s))
    best_example = examples.iloc[index, :].to_frame().T

    return best_example


def validate(valid, example, config):
    validation_data = valid.copy()
    answer = []
    sentiment_mapping = {
        'great': 4,
        'good': 3,
        'okay': 2,
        'bad': 1,
        'terrible': 0
    }
    if config['method'] != 'Default':
        example_data = example.copy()
        example_data['label'] = example_data['label'].map({v: k for k, v in sentiment_mapping.items()})
        example = ''
        for _, row in example_data.iterrows():
            example += f'Comment: {row["comment"]} Label: {row["label"]}\n'
    for i in range(0, len(validation_data), config['batch_size']):
        batch = validation_data.iloc[i:i + config['batch_size'], 1].tolist()
        if config['method'] != 'Default':
            prompt = (f'Please classify whether the sentiment of each of the following movie comments is '
                      f'great, good, okay, bad or terrible.\n '
                      f'Answer with the following format: '
                      f'Comment i: @@@ great/good/okay/bad/terrible @@@, do not include your reasons.\n'
                      f'Here are a few examples:\n{example}'
                      f'Now that you have had some experience with this task, classify the following comments:\n')
        else:
            prompt = (f'Please classify whether the sentiment of each of the following movie comments '
                      'is great, good, okay, bad or terrible.\n'
                      'Answer with the following format: Comment i: @@@ great/good/okay/bad/terrible @@@, '
                      'do not include your reasons.\n')
        for idx, comment in enumerate(batch, start=1):
            prompt += f'Comment{idx}: {comment}\n'
        response = config['client'].chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        response_text = response.choices[0].message.content
        response_text = response_text.strip().split('\n')
        responses = [response.split('@@@ ')[1].split(' @@@')[0] for response in response_text]
        for idx in range(len(responses)):
            extracted_answer = responses[idx].strip()
            answer.append(extracted_answer)
        print(f'Current Progress: {round(((i + config["batch_size"]) / len(validation_data)) * 100, 1)}%')
    answer = [sentiment_mapping[sentiment] for sentiment in answer]
    validation_data['pred'] = answer
    acc = accuracy_score(validation_data['label'], validation_data['pred'])

    return acc


def div_guided_search(train, valid, config):
    examples = get_filtered_data(train, config)
    permutations = []
    print(f'Starting Diversity Guided Search on {len(examples)} examples with {config["iter_num"]} '
          f'iterations and {config["beam_size"]} beams.')
    for i in range(config["iter_num"]):
        if i != 0:
            permutations = [df for df, accuracy in permutations]
        else:
            permutations = [examples.sample(n=10).reset_index(drop=True) for _ in range(config["beam_size"])]
        epsilon = []
        for j in range(config["beam_size"]):
            E = permutations[j]
            for b in range(config['substitution_size']):
                sample = E.sample(1)
                E_star = E.drop(sample.index).reset_index(drop=True)
                sample_new = sample_calculator(examples, E, config)
                E_star = pd.concat([E_star, sample_new])
                epsilon.append(E_star)
            for b in range(config["beam_size"] - config['substitution_size']):
                E_star = E.sample(frac=1).reset_index(drop=True)
                epsilon.append(E_star)
        print(f'Iteration {i+1} Searching for best permutations:')
        permutations = []
        for idx, df in enumerate(epsilon):
            print(f"Iteration {i+1}/{config['iter_num']} Validating dataframe {idx + 1}/{len(epsilon)}:")
            accuracy = validate(valid, df, config)
            permutations.append((df, accuracy))
        permutations.sort(key=lambda x: x[1], reverse=True)
        permutations = permutations[:config["beam_size"]]
        print(f'The optimal validation accuracy for iteration{i+1} is {permutations[0][1]}')
    permutations[0][0].to_csv('SST5_DivSearch_Examples.csv', index=False)
    print('The final optimal validation accuracy is:', permutations[0][1])

    return permutations[0][0]


def get_search_result(train, valid, config):
    file_name = 'SST5_DivSearch_Examples.csv'
    if os.path.exists(file_name):
        print(f"File {file_name} found. Reading CSV...")
        result = pd.read_csv(file_name)
    else:
        print(f"File {file_name} not found. Running div_guided_search...")
        result = div_guided_search(train, valid, config)

    return result