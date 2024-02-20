import json
import ast
import re
import os
work_dir = os.environ["WORK_DIR"]

def encoding(traj):
    solver_tags = re.findall(r'<solver> (human|agent)', traj)
    encoded_solver_tags = ['1' if tag == 'human' else '0' for tag in solver_tags]
    encoded_trajectory = ''.join(encoded_solver_tags)
    return encoded_trajectory

class localgpt:
    def __init__(self, data) -> None:
        if data == 'train':
            filename = work_dir + '/test_data/localgpt_train250.jsonl'
        elif data == 'dev':
            filename = work_dir + '/test_data/localgpt_test100.jsonl'
        with open(filename, 'r') as file:
            loaded_data = json.load(file)
        self.converted_data = {}

        for key, value in loaded_data.items():
            info = {}
            converted_key = ast.literal_eval(key)
            if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                converted_value = ast.literal_eval(value)
                info['trajectory'] = converted_value[0]
                info['em'] = converted_value[1]
                info['f1'] = converted_value[2]
            else:
                converted_value = value
                info['trajectory'] = converted_value

            self.converted_data[converted_key] = info
            
    def get_response(self, episode_idx, trajectory):
        encoded_trajectory = encoding(trajectory)
        return self.converted_data[(episode_idx, encoded_trajectory)]


