import argparse
import os
import json
import re
import copy
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from prompt import SYSTEM_PROMPT, JUDGE_PROMPT, SYSTEM_PROMPT_CN, JUDGE_PROMPT_CN
from google import genai
from google.genai import types
import anthropic

load_dotenv(override=True)


class BrowsecampEval:
    def __init__(self, args):
        self.args = args
        if self.args.model in ["Qwen2.5-Max", "QwQ-32B", "Qwen3-235B-thinking", "Qwen3-235B-no-thinking"]:
            self.args.max_workers = 3 # Qwen2.5-Max may have a maximum request limit (or rate limit) depending on the specific API or service provider's policy.
        
        if self.args.model in ['GPT-4o', 'O1']:
            self.client = OpenAI(api_key=os.getenv('openai_api_key'))
        elif self.args.model == 'O4-mini':
            self.client = OpenAI(api_key=os.getenv('openai_v2_api_key'), base_url=os.getenv('openai_v2_base_url'))
            self.args.max_workers = 5
        elif self.args.model in ['DeepSeek-R1', 'DeepSeek-V3']:
            self.client = OpenAI(api_key=os.getenv('deepseek_api_key'), base_url=os.getenv('deepseek_base_url'))
        elif self.args.model in ['Qwen2.5-72B-Instruct', 'QwQ-32B', 'Qwen2.5-Max', "Qwen3-235B-thinking", "Qwen3-235B-no-thinking"]:
            self.client = OpenAI(api_key=os.getenv('qwen_api_key'), base_url=os.getenv('qwen_base_url'))
        elif self.args.model == 'Llama4':
            self.client = OpenAI(api_key=os.getenv('llama_api_key'), base_url=os.getenv('llama_base_url'))
        elif self.args.model in ['Gemini2.5-Pro', 'Gemini2.0-Flash']:
            self.client = genai.Client(api_key=os.getenv('gemini_api_key'))
        elif self.args.model in ['Claude3.7-think', 'Claude3.5-Sonnet']:
            self.client = anthropic.Anthropic(api_key=os.getenv('claude_api_key'))
            self.args.max_workers = 3 # Claude enforces a rate limit of 80,000 tokens per minute
        
        self.eval_client = OpenAI(api_key=os.getenv('openai_api_key'))
        
    def get_remote_response(self, messages, eval=False):
        model_name_dict = {
            "GPT-4o": "gpt-4o",
            "O1": "o1",
            "O4-mini": "o4-mini",
            "Claude3.7-think": "claude-3-7-sonnet-20250219",
            "Claude3.5-Sonnet": "claude-3-5-sonnet-20240620",
            "Gemini2.5-Pro": "gemini-2.5-pro-preview-03-25",
            "Gemini2.0-Flash": "models/gemini-2.0-flash",
            "Qwen2.5-Max": "qwen-max-2025-01-25",
            "Qwen2.5-72B-Instruct":"qwen2.5-72b-instruct",
            "DeepSeek-R1": "deepseek-r1-250120",
            "DeepSeek-V3": "deepseek-v3-250324",
            "QwQ-32B":"qwq-32b",
            "Llama4": "accounts/fireworks/models/llama4-maverick-instruct-basic",
            "Qwen3-235B-thinking": "qwen3-235b-a22b",
            "Qwen3-235B-no-thinking": "qwen3-235b-a22b",
        }
        if eval:
            model_name = "gpt-4o"
            response = self.eval_client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False,
                temperature=0.6,
                top_p=0.95
            )
            return response.choices[0].message.content
        else:
            try:
                model_name = model_name_dict[self.args.model]
                if self.args.model in ["Gemini2.5-Pro", "Gemini2.0-Flash"]:
                    contents = [
                        types.Content(role="model", parts=[types.Part(text=messages[0]["content"])]),
                        types.Content(role="user", parts=[types.Part(text=messages[1]["content"])])
                    ]
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config={
                            "temperature": 0.6,
                            "top_p": 0.95,
                        }
                    )
                    return response.text, response.usage_metadata.total_token_count
                elif self.args.model == "DeepSeek-R1":
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False,
                        temperature=0.6,
                        top_p=0.95
                    )
                    return "<think>" + response.choices[0].message.reasoning_content + \
                            "</think>\n<answer>" + response.choices[0].message.content + "</answer>", response.usage.total_tokens
                elif self.args.model == "Claude3.7-think":
                    with self.client.beta.messages.stream( 
                        model=model_name,
                        system=SYSTEM_PROMPT_CN,
                        thinking={"type": "enabled", "budget_tokens": 32000},
                        messages=messages[1:],
                        max_tokens=128000,
                        betas=["output-128k-2025-02-19"],
                        temperature=1.0
                    ) as stream:
                        response_text = "<think>"
                        latest_status = "thinking"
                        for event in stream: 
                            if event.type == "text":
                                if latest_status == "thinking":
                                    latest_status = "text"
                                    response_text += f"</think>\n<answer>{event.text}"
                                else:
                                    response_text += event.text
                            elif event.type == "thinking":
                                response_text += f"{event.thinking}"
                            if event.type == 'message_stop':
                                input_tokens = event.message.usage.input_tokens
                                output_tokens = event.message.usage.output_tokens
                                total_tokens = input_tokens + output_tokens
                    return response_text.strip() + "</answer>", total_tokens
                elif self.args.model == "Claude3.5-Sonnet":
                    response = self.client.messages.create(
                        model=model_name,
                        system=SYSTEM_PROMPT_CN,
                        messages=messages[1:],
                        max_tokens=8000,
                        temperature=0.6,
                        top_p=0.95,
                    )
                    return response.content[0].text, response.usage.input_tokens + response.usage.output_tokens
                elif self.args.model == "Qwen3-235B-no-thinking":
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=True,
                        temperature=0.6,
                        top_p=0.95,
                        stream_options={"include_usage": True},
                        extra_body={"enable_thinking": False}
                    )
                    response_text = ""
                    latest_status = "think"
                    for chunk in response:
                        if chunk.choices == []:
                            total_tokens = chunk.usage.total_tokens
                            continue
                        if chunk.choices[0].delta.content is not None:
                            if latest_status == "think":
                                latest_status = "answer"
                                response_text += chunk.choices[0].delta.content
                            else:
                                response_text += chunk.choices[0].delta.content
                        else:
                            response_text += chunk.choices[0].delta.reasoning_content
                    return response_text.strip(), total_tokens
                elif self.args.model in ["QwQ-32B", "Qwen3-235B-thinking"]:
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=True,
                        temperature=0.6,
                        top_p=0.95,
                        stream_options={"include_usage": True},
                        extra_body={"enable_thinking": True if self.args.model != "QwQ-32B" else False}
                    )
                    response_text = "<think>"
                    latest_status = "think"
                    for chunk in response:
                        if chunk.choices == []:
                            total_tokens = chunk.usage.total_tokens
                            continue
                        if chunk.choices[0].delta.content is not None:
                            if latest_status == "think":
                                latest_status = "answer"
                                response_text += "</think>\n<answer>" + chunk.choices[0].delta.content
                            else:
                                response_text += chunk.choices[0].delta.content
                        else:
                            response_text += chunk.choices[0].delta.reasoning_content
                    return response_text.strip() + "</answer>", total_tokens
                elif self.args.model == "O1":
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False,
                        timeout=3000
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False,
                        temperature=0.6,
                        top_p=0.95
                    )
                return response.choices[0].message.content, response.usage.total_tokens
            except Exception as e:
                print(f"question: {messages[1]['content']} and problem: {e}")
                return "", 0
        # print(f"prompt_tokens:{response.usage.prompt_tokens}, completion_tokens:{response.usage.completion_tokens}, total_tokens:{response.usage.total_tokens}")
        
    
    def get_multi_infer_response(self):
        # 1 build_infer_chat_data
        chat_datas = []
        with open(self.args.input_file_path, 'r') as f:
            querys = json.load(f)
        for query in querys:
            chat_data = {}
            message = [
                {"role": "system", "content": SYSTEM_PROMPT_CN},
                {"role": "user", "content": query["Question"]},
                ]
            chat_data['messages'] = message
            chat_data['question'] = query["Question"]
            chat_data['answer'] = query["Answer"]
            chat_datas.append(chat_data)
        
        # 2 get response
        # 3 save
        predict_infer_file_dir = os.path.join(self.args.predict_infer_file_dir, self.args.model)
        os.makedirs(predict_infer_file_dir, exist_ok=True)
        with open(os.path.join(predict_infer_file_dir, 'infer.jsonl'), 'a') as f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
                futures = [executor.submit(self.get_remote_response, chat_data['messages']) for chat_data in chat_datas]
                for i, future in tqdm(enumerate(futures), desc="Generating infer responses", total=len(futures)):
                    response, total_tokens = future.result()
                    if response is None:
                        response = "None"
                    if total_tokens is None:
                        total_tokens = 0
                    pattern = r"""
                        \*{0,2}Explanation\*{0,2}\s*?:\s*(.*?)\n
                        \*{0,2}Exact\sAnswer\*{0,2}\s*?:\s*(.*?)\n
                        \*{0,2}Confidence\*{0,2}\s*?:\s*(.*?)$
                    """
                    matches = re.search(pattern, response, re.DOTALL | re.VERBOSE)
                    if matches:
                        explanation = matches.group(1).strip()
                        exact_answer = matches.group(2).strip()
                        confidence = matches.group(3).strip()
                    else:
                        explanation, exact_answer, confidence = "", "", ""
                    result = {"question": chat_datas[i]['question'], "answer": chat_datas[i]['answer'], "response": response,
                        "explanation": explanation, "exact_answer": exact_answer, "confidence": confidence, "total_tokens": total_tokens}
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
    def generate_infer_eval(self):
        # 1 load data
        with open(self.args.input_file_path, 'r') as f:
            raw_datas = json.load(f)
        raw_datas_copy = copy.deepcopy(raw_datas)
        
        infer_datas = []
        with open(os.path.join(self.args.predict_infer_file_dir, self.args.model, 'infer.jsonl'), 'r') as f:
            for line in f:
                infer_datas.append(json.loads(line))
        
        # 2 judge answer
        messages = []
        for i, infer_data in enumerate(infer_datas):
            message = [
                {"role": "system", "content": "you are a helpful assistant!"},
                {"role": "user", "content": JUDGE_PROMPT_CN.format(question=infer_data['question'], response=infer_data['response'], correct_answer=infer_data['answer'])},
            ]
            messages.append(message)
        
        # 3 save
        eval_infer_file_dir = os.path.join(self.args.eval_infer_file_dir, self.args.model)
        output_infer_file_dir = os.path.join(self.args.output_infer_file_dir, self.args.model)
        os.makedirs(eval_infer_file_dir, exist_ok=True)
        os.makedirs(output_infer_file_dir, exist_ok=True)
        with open(os.path.join(eval_infer_file_dir, 'infer.jsonl'), 'a') as eval_f, \
             open(os.path.join(output_infer_file_dir, 'infer.jsonl'), 'a') as final_f:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
                futures = [executor.submit(self.get_remote_response, message, True) for message in messages]
                for i, future in tqdm(enumerate(futures), desc="Generating infer eval responses", total=len(futures)):
                    response = future.result()
                    pattern = r"""
                        \*{0,2}extracted_final_answer\*{0,2}\s*?:\s*(.*?)\n
                        \*{0,2}reasoning\*{0,2}\s*:\s*?(.*?)\n
                        \*{0,2}correct\*{0,2}\s*:\s*?(.*?)\n
                        \*{0,2}confidence\*{0,2}\s*?:\s*(.*?)$
                        """
                    matches = re.search(pattern, response, re.DOTALL | re.VERBOSE)

                    if matches:
                        model_extracted_answer = matches.group(1).strip()
                        reasoning = matches.group(2).strip()
                        is_correct = matches.group(3).strip()
                        model_extracted_confidence = matches.group(4).strip()
                    else:
                        model_extracted_answer, reasoning, is_correct, model_extracted_confidence = "", "", "", ""
                    assert raw_datas_copy[i]['Question'] == infer_datas[i]['question']
                    eval_result = {
                        "model_extracted_answer": model_extracted_answer,
                        "model_prediction": infer_datas[i]["response"],
                        "is_correct": is_correct,
                        "model_extracted_confidence": model_extracted_confidence
                    }
                    chat_data = {
                        "question": raw_datas_copy[i]['Question'],
                        "answer": raw_datas_copy[i]['Answer'],
                        "response": response,
                        "model_extracted_answer": model_extracted_answer,
                        "reasoning": reasoning,
                        "is_correct": is_correct,
                        "model_extracted_confidence": model_extracted_confidence
                    }
                    raw_datas_copy[i]["eval_result"] = [eval_result]
                    eval_f.write(json.dumps(chat_data, ensure_ascii=False) + '\n')
                    final_f.write(json.dumps(raw_datas_copy[i], ensure_ascii=False) + '\n')
            
    def eval_infer(self):
        # 1 get response
        self.get_multi_infer_response()
        # 2 generate eval
        self.generate_infer_eval()

    def run(self):
        self.eval_infer()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # open-source:  DeepSeek-V3 DeepSeek-R1 Qwen2.5-72B-Instruct QwQ-32B Llama4 Qwen3-235B-thinking Qwen3-235B-no-thinking
    # close-source: GPT-4o O1 O4-mini Claude3.5-Sonnet Claude3.7-think Gemini2.0-Flash Gemini2.5-Pro Qwen2.5-Max
    parser.add_argument('--model', type=str, default="DeepSeek-R1")
    parser.add_argument('--input_file_path', type=str, default=f"raw_data/test.json")
    parser.add_argument('--predict_infer_file_dir', type=str, default=f"predict_data")
    parser.add_argument('--eval_infer_file_dir', type=str, default=f"eval_data")
    parser.add_argument('--output_infer_file_dir', type=str, default=f"output_data")
    parser.add_argument('--max_workers', type=int, default=10)
    args = parser.parse_args()
    
    eval = BrowsecampEval(args)
    eval.run()
    
