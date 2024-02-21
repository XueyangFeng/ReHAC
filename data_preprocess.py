import argparse
import json
import re
from collections import defaultdict


def process_data(file_name, lambda_, output_file_name):
    episodes = []
    with open(file_name, "r") as f:
        for line in f:
            episode = json.loads(line)
            trajectory = episode['trajectory']
            segments = trajectory.split("<solver>")
            current_state = ""
            budget = 0

            for i, segment in enumerate(segments):
                if i == 0:
                    continue  
                current_state += segments[i - 1] + "<solver>"
                current_state = current_state.strip()
                action = 0 if "<solver>agent" in segment else 1
                if action == 1:
                    budget += lambda_
                reward = episode['f1'] - budget
                episodes.append({"state": current_state, "action": action, "reward": reward})


    sar_rewards = defaultdict(lambda: {"reward_sum": 0, "count": 0})
    for sar in episodes:
        key = (sar["state"], sar["action"])
        sar_rewards[key]["reward_sum"] += sar["reward"]
        sar_rewards[key]["count"] += 1


    combined_state_data = []
    for (state, action), data in sar_rewards.items():
        if len(state) <= 5041:  
            combined_state_data.append({
                "state": state,
                "action": action,
                "reward": data["reward_sum"] / data["count"],
                "count": data["count"]
            })

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