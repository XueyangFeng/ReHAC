import argparse
import json
import re
from collections import defaultdict

def process_data(input_file_name, lambda_, output_file_name):
    with open(input_file_name, "r") as f:
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
                if 'human' in solver_regex.findall("<solver>" + segment):
                    budget += lambda_

            for i in range(1, len(segments)):
                current_state += segments[i - 1] + "<solver>"
                current_state = current_state.strip()
                action = 0 if 'agent' in solver_regex.findall("<solver>" + segments[i]) else 1
                reward = episode['f1'] - budget
                sar_list_episode.append({"state": current_state, "action": action, "reward": reward})

            episodes.append(sar_list_episode)

    sar_list_all = [sar for episode in episodes for sar in episode]
    sar_rewards = defaultdict(lambda: {"reward_sum": 0, "count": 0})
    for sar in sar_list_all:
        key = (sar["state"], sar["action"])
        sar_rewards[key]["reward_sum"] += sar["reward"]
        sar_rewards[key]["count"] += 1


    state_action = {}
    for (state, action), info in sar_rewards.items():
        if len(state) <= 5041:
            if state not in state_action:
                state_action[state] = {'action1': action, 'reward1': info["reward_sum"] / info["count"], 'count1': info["count"]}
            else:
                state_action[state]['action2'] = action
                state_action[state]['reward2'] = info["reward_sum"] / info["count"]
                state_action[state]['count2'] = info["count"]

    combined_state_data = []
    for state, data in state_action.items():
        if 'action2' in data:
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
            combined_state_data.extend([combined_first_entry, combined_second_entry])


    with open(output_file_name, 'w') as f:
        for entry in combined_state_data:
            json.dump(entry, f)
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('file_name', type=str, help='Input file name')
    parser.add_argument('lambda_', type=float, help='Lambda value')
    parser.add_argument('output_file_name', type=str, help='Output file name')

    args = parser.parse_args()

    process_data(args.file_name, args.lambda_, args.output_file_name)
