"""
    数据格式按照<solver>human开始造
""" 
import openai
import random
import time
import wikienv, wrappers
import json
import requests
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="train")  # dev 表示验证集 react论文选取500条验证集测试 train表示训练集
env = wrappers.LoggingWrapper(env)
random.seed(0)  # seed for human help introduce

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatgpt(prompt, stop=["\n"]):
    openai.api_key = "xx"
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0613",
      messages=messages,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response.choices[0]["message"]["content"]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt4(prompt, stop=["\n"]):
    openai.api_key = "xx"
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
      model="gpt-4-0613",
      messages=messages,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response.choices[0]["message"]["content"]


def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1


folder = './prompts/'
prompt_file = 'prompts_gpt35.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']

instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""

webthink_prompt = instruction + webthink_examples

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    prompt += question + "\n"
    if to_print:
        print(idx, question)
    output = ""
    output += question + "\n"
    n_calls, n_badcalls = 0, 0
    
    agent_solver = "<solver> agent\n"
    human_solver = "<solver> human\n"

    for i in range(1, 10):   # ReAct:8  My:10 so the agent will most introduce 2 human help
        if random.random() > 0.4:   # solver: agent
            n_calls += 1
            thought_action = chatgpt(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except:
                print('ohh...', thought_action)
                n_badcalls += 1
                n_calls += 1
                thought = thought_action.strip().split('\n')[0]
                action = chatgpt(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
            obs, r, done, info = step(env, action[0].lower() + action[1:])
            obs = obs.replace('\\n', '')
            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            prompt += step_str
            output += agent_solver + step_str
            if to_print:
                print(step_str)
            if done:
                break
        else:
            thought_action = gpt4(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except:
                print('ohh...', thought_action)
                thought = thought_action.strip().split('\n')[0]
                action = gpt4(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
            obs, r, done, info = step(env, action[0].lower() + action[1:])
            obs = obs.replace('\\n', '')
            step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            prompt += step_str
            output += human_solver + step_str
            if to_print:
                print(step_str)
            if done:
                break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info, output


def main():
    # idxs = list(range(7405))     # dev set
    idxs = list(range(90447)) # train data set
    random.Random(233).shuffle(idxs)
    ##
    f1_scores = []
    rs = []
    infos = []
    old_time = time.time()
    for i in idxs[10:100]:
        if i in [0, 2, 27, 33, 35, 71103]:
            continue
        # prompt instance index 71103 2 35 33 0 27
        r, info, trajectory = webthink(i, to_print=True)
        rs.append(info['em'])
        f1_scores.append(info['f1'])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print('-----------')
        print()
        print(f"EM:\n{rs}")
        print(f"F1 Score:\n{f1_scores}")
        
        # Load and Write trajectory
        with open("./traj_solver_100.json", "r", encoding="utf-8") as json_file:
            old_data = json.load(json_file)
            old_data.update({i: trajectory})
        with open("./traj_solver_100.json", "w", encoding="utf-8") as json_file:
            json.dump(old_data, json_file)
    
    # Load and Write EM and F1 Score
    with open('./score_solver_100.txt', "a", encoding="utf-8") as txt_file:
        txt_file.write("\nEM\n" + str(rs))
        txt_file.write("\nF1\n" + str(f1_scores))

if __name__ == "__main__":
    main()