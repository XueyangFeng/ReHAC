import json
import ast
import re
import os
work_dir = os.environ["WORK_DIR"]

#输入一整条轨迹，输出编码（01）后的轨迹
def encoding(traj):
    # 提取轨迹中的 <solver> 标签部分
    solver_tags = re.findall(r'<solver> (human|agent)', traj)
    # 将提取出的标签转换为 "1" 和 "0"
    encoded_solver_tags = ['1' if tag == 'human' else '0' for tag in solver_tags]
    # 将列表转换为字符串
    encoded_trajectory = ''.join(encoded_solver_tags)
    return encoded_trajectory

class localgpt:
    def __init__(self, data) -> None:
        if data == 'train':
            # 指定要读取的JSON文件名
            filename = work_dir + '/test_data/localgpt_train250.jsonl'
        elif data == 'dev':
            filename = work_dir + '/test_data/localgpt_test100.jsonl'
        #filename = '/root/human-agent-xueyang/test_data/localgpt.jsonl'

        # 读取JSON文件
        with open(filename, 'r') as file:
            loaded_data = json.load(file)


        # 转换字符串键回元组
        #self.converted_data = {ast.literal_eval(key): ast.literal_eval(value) for key, value in loaded_data.items()}
        self.converted_data = {}

        

        for key, value in loaded_data.items():
            info = {}
            converted_key = ast.literal_eval(key)
            # 检查 value 是否是元组格式
            if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                converted_value = ast.literal_eval(value)
                info['trajectory'] = converted_value[0]
                info['em'] = converted_value[1]
                info['f1'] = converted_value[2]
            else:
                # 如果 value 不是元组格式，直接使用原始字符串
                converted_value = value
                info['trajectory'] = converted_value

            self.converted_data[converted_key] = info
            


    def get_response(self, episode_idx, trajectory):
        encoded_trajectory = encoding(trajectory)
        return self.converted_data[(episode_idx, encoded_trajectory)]


