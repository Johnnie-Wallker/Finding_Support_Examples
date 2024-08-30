import pandas as pd
from Diversity_Guided_Search import get_search_result, validate
pd.options.mode.copy_on_write = True


def run_llm(config):
    data_path = f'full_train_data/{config["data"]}'
    data = pd.read_csv(f'{data_path}/test.csv', header=None)
    train = pd.read_csv(f'{data_path}/train.csv', header=None)
    data.columns = ['label', 'comment']
    train.columns = ['label', 'comment']
    if config['method'] == 'LENS':
        valid = train.sample(config['valid_size'], random_state=1)
        train = train.drop(valid.index)
        examples = get_search_result(train, valid, config)
    elif config['method'] == 'Random':
        examples = train.sample(n=10, random_state=1)
    elif config['method'] == 'RandomByCategory':
        examples_per_label = []
        for label in train['label'].unique():
            label_examples = train[train['label'] == label]
            sampled_examples = label_examples.sample(n=2, random_state=1)
            examples_per_label.append(sampled_examples)
        examples = pd.concat(examples_per_label)
        examples = examples.sort_values('label')
    else:
        examples = None
    acc = validate(data, examples, config)

    print(f'准确率为：{round(acc, 3)}')