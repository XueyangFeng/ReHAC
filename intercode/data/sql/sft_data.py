
import json
from collections import defaultdict

delta = 0.08
DATA = 100

state_action = {}
with open(f'./sample_count_data{DATA}_budget{delta}_len_limit.jsonl', 'r') as f:
    for line in f:
        sar = json.loads(line)
        state = sar['state']
        action = sar['action']
        reward = sar['reward']

        if state not in state_action:
            state_action[state] = {'action1': action,  'reward1': reward}
            state_action[state]['flag'] = False
            state_action[state]['count1'] = sar['count']
        else:
            state_action[state]['action2'] = action
            state_action[state]['reward2'] = reward
            state_action[state]['count2'] = sar['count']




combined_state_data = []



for state, data in state_action.items():
    if 'action2' in data and 'reward2' in data:
        if data['reward1']>=data['reward2']:
            combined_entry = {
                'state': state,
                'action': data['action1'],
                'count': data['count1']
            }
        else:
            combined_entry = {
                'state': state,
                'action': data['action2'],
                'count': data['count2']
            }
        combined_state_data.append(combined_entry)
    else:
        continue


with open(f'sft_sample_count_{delta}.jsonl', 'w') as f:
    for entry in combined_state_data:
        json.dump(entry, f)
        f.write('\n')