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
# env = wrappers.HotPotQAWrapper(env, split="dev")  # dev 表示验证集 react论文选取500条验证集测试
env = wrappers.HotPotQAWrapper(env, split="train")
env = wrappers.LoggingWrapper(env)
random.seed(0)  # seed for human help introduce

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt4(prompt, stop=["\n"]):
    openai.api_key = "xxx"
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

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

folder = './prompts/'
prompt_file = 'prompts_gpt35_human.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple']

agent_instruction = """Imagine you are an agent. Please help me solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Human can step in and provide help after each step. Your output and human help follows the following format:
<Agent>: your Thought, Action and Observation.\n<Human>: human Thought, Action and Observation, or None.\n
Here are some examples.
"""

human_instruction = """Imagine you are a helpful human. Please help me solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
You should step in and provide help now. Please think carefully and independently. Agent output and your help follows the following format:
<Agent>: agent Thought, Action and Observation.\n<Human>: your Thought, Action and Observation.\n
Here are some examples.
"""

def webthink(idx=None, prompt=webthink_examples, to_print=True):
    question = env.reset(idx=idx)
    prompt += question + "\n"
    if to_print:
        print(idx, question)
    trajectory = ""
    trajectory += question + "\n"
    n_calls, n_badcalls = 0, 0
    i = 1
    while i < 12:  # ReAct:8  My:12 so the agent will most introduce 4 human help
        ######### Introduce Agent help
        n_calls += 1
        thought_action = chatgpt(agent_instruction + prompt + f"<Agent>Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = chatgpt(agent_instruction + prompt + f"<Agent>Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"<Agent>Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        trajectory += step_str
        if to_print:
            print(step_str)

        if done:
            break
        i += 1
        
        ########## Introduct human help ############
        if random.random() < 0.4:  # 40%
            thought_action = gpt4(human_instruction + prompt + f"<Human>Thought {i}:", stop=[f"\nObservation {i}:"])
            try:
                thought, action = thought_action.strip().split(f"\nAction {i}: ")
            except:
                print('ohh...', thought_action)
                thought = thought_action.strip().split('\n')[0]
                action = gpt4(human_instruction + prompt + f"<Human>Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
            obs, r, done, info = step(env, action[0].lower() + action[1:])
            obs = obs.replace('\\n', '')
            step_str = f"<Human>Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
            prompt += step_str
            trajectory += step_str
            if to_print:
                print(step_str)

            if done:
                break
            i += 1
        else:
            prompt += "<Human>None.\n"
            trajectory += "<Human>None.\n"


    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info, trajectory


def main():
    ## dev set
    # idxs = list(range(7405))
    idxs = list(range(90447)) # train set
    random.Random(233).shuffle(idxs)
    ##
    f1_scores = []
    rs = []
    infos = []
    old_time = time.time()
    # for i in idxs[:500]:
    # all_trajectory = {}
    for i in idxs[5:50]:
        if i in [0, 2, 27, 33, 35, 71103]:
            continue
        r, info, trajectory = webthink(i, to_print=True)
        # all_trajectory[i] = trajectory
        rs.append(info['em'])
        f1_scores.append(info['f1'])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print('-----------')
        print(f"F1 Score:\n{f1_scores}")
        print()

        # Load and Write trajectory
        with open("./traj.json", "r", encoding="utf-8") as json_file:
            old_data = json.load(json_file)
            # old_data.update(all_trajectory)
            old_data.update({i: trajectory})
        with open("./traj.json", "w", encoding="utf-8") as json_file:
            json.dump(old_data, json_file)
    
    # Load and Write EM and F1 Score
    with open('./score.txt', "a", encoding="utf-8") as txt_file:
        txt_file.write("\nEM\n" + str(rs))
        txt_file.write("\nF1\n" + str(f1_scores))
    



if __name__ == "__main__":
    main()