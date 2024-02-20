import os
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

openai.api_key = "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"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def llm(prompt, model, stop=["\n"]):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
      model=model,
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
prompt_file = 'prompts_gpt35_wo.json'
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
    output = ""
    question = env.reset(idx=idx)
    prompt += question + "\n"
    if to_print:
        print(idx, question)
        output += question + "\n"

    n_calls, n_badcalls = 0, 0

    for i in range(1, 8):
        if i == 1 or i == 2:
            continue
        prompt_suffix = "Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?\nThought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.\nAction 1: Search[Pavel Urysohn]\nObservation 1: Pavel Samuilovich Urysohn (February 3, 1898 \u00e2\u0080\u0093 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.Thought 2: The type of work Pavel Urysohn is known for is dimension theory. I need to search Leonid Levin next and find his type of work.\nAction 2: Search[Leonid Levin]\nObservation 2: Leonid Anatolievich Levin (/le\u026a.o\u028a\u02c8ni\u02d0d \u02c8l\u025bv\u026an/ lay-oh-NEED LEV-in; Russian: \u041b\u0435\u043e\u043d\u0438\u0301\u0434 \u0410\u043d\u0430\u0442\u043e\u0301\u043b\u044c\u0435\u0432\u0438\u0447 \u041b\u0435\u0301\u0432\u0438\u043d; Ukrainian: \u041b\u0435\u043e\u043d\u0456\u0301\u0434 \u0410\u043d\u0430\u0442\u043e\u0301\u043b\u0456\u0439\u043e\u0432\u0438\u0447 \u041b\u0435\u0301\u0432\u0456\u043d; born November 2, 1948) is a Soviet-American mathematician and computer scientist.. He is known for his work in randomness in computing, algorithmic complexity and intractability, average-case complexity,[1] foundations of mathematics and computer science, algorithmic probability, theory of computation, and information theory. He obtained his master's degree at Moscow University in 1970 where he studied under Andrey Kolmogorov and completed the Candidate Degree academic requirements in 1972.[2]. He and Stephen Cook independently discovered the existence of NP-complete problems. This NP-completeness theorem, often called the Cook\u2013Levin theorem, was a basis for one of the seven Millennium Prize Problems declared by the Clay Mathematics Institute with a $1,000,000 prize offered.\n"
        prompt = webthink_prompt + prompt_suffix
        print("=======prompt=====")
        print(prompt)
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", model="gpt-3.5-turbo", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", model="gpt-3.5-turbo", stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            output += step_str
            print(step_str)
            with open("./traj.json", "w", encoding="utf-8") as jsonfile:
                json.dump({"answer": prompt_suffix + step_str}, jsonfile);raise
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
    rs = []
    infos = []
    old_time = time.time()
    for i in idxs[:500]:
        # prompt instance index 71103 2 35 33 0 27
        r, info, trajectory = webthink(i, to_print=True)
        with open("./traj.json", "w", encoding="utf-8") as file:
            json.dump({f"{i}": trajectory}, file)
        rs.append(info['em'])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print('-----------')
        print()

if __name__ == "__main__":
    main()