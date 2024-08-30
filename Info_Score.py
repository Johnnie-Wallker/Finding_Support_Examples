import pandas as pd
pd.options.mode.copy_on_write = True


def info(example, test, config):
    test_data = test.copy()
    test_data['pred1'] = model_predict(test_data, example, config)
    test_data['pred2'] = model_predict(test_data, None, config)
    test_data['diff1'] = abs(test_data['pred1'] - test_data['label'])
    test_data['diff2'] = abs(test_data['pred2'] - test_data['label'])
    df = test_data['diff2'] - test_data['diff1']
    return df


def model_predict(test, example, config):
    sentiment_mapping = {
        'great': 4,
        'good': 3,
        'okay': 2,
        'bad': 1,
        'terrible': 0
    }
    if example is not None:
        reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
        example['label'] = reverse_sentiment_mapping[example['label']]
        prompt = (f'Please classify whether the sentiment of each of the following movie comments is '
                  f'great, good, okay, bad or terrible.\n '
                  f'Answer with the following format: '
                  f'Comment i: @@@ great/good/okay/bad/terrible @@@, do not include your reasons.\n'
                  f'Here is an example comment:\n'
                  f'Comment: {example["comment"]} Label: {example["label"]}\n'
                  f'Now that you have had some experience with this, classify the following comments:\n')
        for idx, comment in enumerate(test.iloc[:, 1].tolist(), start=1):
            prompt += f'Comment{idx}: {comment}\n'
    else:
        prompt = (f'Please classify whether the sentiment of each of the following movie comments is '
                  f'great, good, okay, bad or terrible.\n '
                  f'Answer with the following format: '
                  f'Comment i: @@@ great/good/okay/bad/terrible @@@, do not include your reasons.\n'
                  f'Classify the following comments:\n')
        for idx, comment in enumerate(test.iloc[:, 1].tolist(), start=1):
            prompt += f'Comment{idx}: {comment}\n'
    response = config['client'].chat.completions.create(
        model=config['model'],
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    response_text = response.choices[0].message.content
    response_text = response_text.strip().split('\n')
    responses = [response.split('@@@ ')[1].split(' @@@')[0] for response in response_text]
    answer = []
    for idx in range(len(responses)):
        extracted_answer = responses[idx].strip()
        answer.append(extracted_answer)
    answer = [sentiment_mapping[sentiment] for sentiment in answer]

    return answer
