
import json
from collections import defaultdict

delta = 0.04
DATA = 100

# 用来存储每个state的最高reward及对应的action
state_action = {}
# 从jsonl文件中读取数据
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

        """
        if reward < state_best_action[state]['reward']:
            state_best_action[state]['reward'] = state_best_action[state]['reward'] - reward
        else:
            state_best_action[state]['reward'] = reward - state_best_action[state]['reward']
            state_best_action[state]['action'] = action
        state_best_action[state]['flag'] = True
        """



# 准备用于写入新JSONL文件的数据
combined_state_data = []



for state, data in state_action.items():
    # 确保 action2 和 reward2 存在
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
        # 如果只有一个 action，跳过这个 state
        continue

"""
for state, data in state_action.items():

    combined_first_entry = {
        'state': state,
        'action': data['action1'] ,
        'reward': (data['reward1'] - data['reward2'])/2,
        'count': data['count1']
    }
    combined_second_entry = {
        'state': state,
        'action': data['action2'] ,
        'reward': (data['reward2'] - data['reward1'])/2,
        'count': data['count2']
    }
    combined_state_data.append(combined_first_entry)
    combined_state_data.append(combined_second_entry)
"""
# 打印部分合并结果（可选）
for entry in combined_state_data[:3]:  # 打印前三个结果来查看格式
    print(entry)

# 将合并的数据写入新的JSONL文件
with open(f'sft_sample_count_{delta}.jsonl', 'w') as f:
    for entry in combined_state_data:
        json.dump(entry, f)
        f.write('\n')