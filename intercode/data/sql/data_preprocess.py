import re
import json

delta = 0.08
DATA = 100


with open(f"./sample_100.jsonl", "r") as f:
    episodes = []

    for line in f:
        episode = json.loads(line)
        trajectory = episode['trajectory']

        solver_regex = re.compile(r'\<solver\>[\s\S]*?(\w+)')

        sar_list_episode = []

        segments = trajectory.split("<solver>")
        current_state = ""

        budget = 0

        for segment in segments:
            solver_regex = re.compile(r'\<solver\>[\s\S]*?(\w+)')
            if 'human' in solver_regex.findall("<solver>"+segment):
                budget += delta

        for i in range(1, len(segments)):
            current_state += segments[i - 1] + "<solver>"
            current_state = current_state.strip()
            solver_regex = re.compile(r'\<solver\>[\s\S]*?(\w+)')

            action = 0 if 'agent' in solver_regex.findall("<solver>"+segments[i]) else 1

            reward = episode['f1'] - budget


            sar_list_episode.append({"state": current_state, "action": action, "reward": reward })

        episodes.append(sar_list_episode)



from collections import defaultdict


sar_list_all = [sar for episode in episodes for sar in episode]

sar_rewards = defaultdict(lambda: {"reward_sum": 0, "count": 0})

for sar in sar_list_all:
  key = (sar["state"], sar["action"])
  sar_rewards[key]["reward_sum"] += sar["reward"]
  sar_rewards[key]["count"] += 1


merged_sar_list = [{"state": state, "action": action, "reward": info["reward_sum"] / info["count"], "count": info["count"]} for (state, action), info in sar_rewards.items()]

print(merged_sar_list[3])

with open(f'./sample_count_data{DATA}_budget{delta}.jsonl', 'w') as f:
    for entry in merged_sar_list:
        json.dump(entry, f)
        f.write('\n')


with open(f'./sample_count_data{DATA}_budget{delta}.jsonl', 'r') as file, open(f'./sample_count_data{DATA}_budget{delta}_len_limit.jsonl', 'a') as output_file:
    for line in file:
        # 解析JSON
        data = json.loads(line)

        entry = json.loads(line)
        trajectory = entry.get('state', '') 
        if len(trajectory) <= 5041:
            output_file.write(json.dumps(data) + '\n')


import json
from collections import defaultdict

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
        combined_first_entry = {
            'state': state,
            'action': data['action1'],
            'reward': (data['reward1'] - data['reward2']) / 2,
            'count': data['count1']
        }
        combined_second_entry = {
            'state': state,
            'action': data['action2'],
            'reward': (data['reward2'] - data['reward1']) / 2,
            'count': data['count2']
        }
        combined_state_data.append(combined_first_entry)
        combined_state_data.append(combined_second_entry)
    else:
        continue

for entry in combined_state_data[:3]:  
    print(entry)


with open(f'advantage_sample_count_{delta}.jsonl', 'w') as f:
    for entry in combined_state_data:
        json.dump(entry, f)
        f.write('\n')



