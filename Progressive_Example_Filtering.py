import pandas as pd
import os
from Info_Score import info


def progressive_example_filter(train, config):
    examples = train.sample(config['data_size'])
    S = train.sample(config['initial_score_size'])
    d = examples.copy()
    iteration = 0
    while len(d) > config['desired_candidate_size']:
        file_dir = (f'SST5_Filtered_Examples/Examples{config["data_size"]}_'
                    f'Keep{config["desired_candidate_size"]}_Test{config["initial_score_size"]}/Iteration {iteration}')
        print(f'Iteration {iteration}: file direction at {file_dir}')
        os.makedirs(file_dir, exist_ok=True)
        if os.path.exists(f'{file_dir}/samples.csv') and os.path.exists(f'{file_dir}/examples.csv'):
            print(f'Iteration {iteration} already completed, proceeding...')
            S = pd.read_csv(f'{file_dir}/samples.csv')
            d = pd.read_csv(f'{file_dir}/examples.csv')
            iteration += 1
        else:
            print(f'Starting iteration {iteration} Examples left: {len(d)} '
                  f'Target examples size: {config["desired_candidate_size"]} Test size: {len(S)}')
            d = d.iloc[:, :2]
            s = []
            print(f'Fetching information scores...')
            for i in range(len(d)):
                s.append(info(d.iloc[i, :], S, config))
                print(f'Current progress: {round(((i+1)/len(d))*100,1)}%')
            s = pd.DataFrame(s)
            d = d.reset_index(drop=True)
            d = pd.concat([d, s], axis=1)
            d['info'] = d.iloc[:, 2:].sum(axis=1)
            d = d.sort_values(by='info', ascending=False)
            if len(d) / config['progressive_factor'] < config["desired_candidate_size"]:
                d = d.head(config["desired_candidate_size"])
                S.to_csv(f'{file_dir}/samples.csv', index=False)
                d.to_csv(f'{file_dir}/examples.csv', index=False)
                break
            else:
                d = d.head(int((1 / config['progressive_factor']) * len(d)))
            S_new = train.sample(config["initial_score_size"] * (config['progressive_factor'] - 1))
            S = pd.concat([S, S_new])
            S = S.drop_duplicates()
            S.to_csv(f'{file_dir}/samples.csv', index=False)
            d.to_csv(f'{file_dir}/examples.csv', index=False)
            iteration += 1

    d.to_csv('SST5_Filtered_Examples.csv', index=False)

    return d


def get_filtered_data(train, config):
    file_name = 'SST5_Filtered_Examples.csv'
    if os.path.exists(file_name):
        print(f"File {file_name} found. Reading CSV...")
        d = pd.read_csv(file_name)
    else:
        print(f"File {file_name} not found. Running progressive_example_filter...")
        d = progressive_example_filter(train, config)

    return d