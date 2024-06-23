from gpt4_moderator import GPT4Moderator
from gemini_moderator import GeminiModerator
from llama_moderator import LlamaModerator
from openai import OpenAI
import os
from tqdm import tqdm
import pandas as pd
import fire
import re
import google.generativeai as genai
import time

all_categories = [
    'hate',
    'self-harm',
    'sexual',
    'violence',
    'harassment',
    'harmful',
    'illegal',
    'unethical',
    'privacy',
    'deception',
    'none'
]


def extract_text(input_string):
    # Use regular expressions to find the text pattern
    # breakpoint()
    pattern = re.compile(r'\[\[\d+\.\s*(.*?)\]\]')
    match = pattern.search(input_string)
    
    if match:
        return match.group(1)
    else:
        return None

def check_category(model_response):
    assigned_category = extract_text(model_response)
    if assigned_category is None:
        return 'none'
    for category in all_categories:
        if category in assigned_category.lower():
            return category
    else:
        return 'none'


def main(moderator_model, input_file_path, column, merge_result=True):
    """
    moderator_model represents which model to use as the moderator.
    input_file_path represents the input csv file containing prompts.
    column decides which column to check, e.g. either prompt or response
    merge_result decides if the output will be separated into different files or not
    """
    if moderator_model == 'gpt4':
        gpt_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        moderator = GPT4Moderator(gpt_client)
    elif moderator_model == 'gemini-1.5':
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        client = genai.GenerativeModel('gemini-1.5-pro-latest')
        moderator = GeminiModerator(client)
    elif moderator_model == 'llama3':
        TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        client = OpenAI(
            api_key=TOGETHER_API_KEY,
            base_url='https://api.together.xyz/v1',
        )
        moderator = LlamaModerator(client)
    else:
        exit("moderator model not supported!")
    
    input_data = pd.read_csv(input_file_path)[column].to_list()
    input_original_prompts = pd.read_csv(input_file_path)['prompt'].to_list()
    input_file_name = input_file_path.split("/")[-1]
    output_root = "./" 
    output_file_toxic = output_root + input_file_name[:-4] + "_toxic_moderation_{}_{}.csv".format(moderator_model, column)
    output_file_non_toxic = output_root + input_file_name[:-4] + "_non_toxic_moderation_{}_{}.csv".format(moderator_model, column)
    output_file_failed = output_root + input_file_name[:-4] + "_moderation_failed_{}_{}.csv".format(moderator_model, column)
    output_file_merged = output_root + input_file_name[:-4] + "_moderation_merged_{}_{}.csv".format(moderator_model, column)


    overall_stats = dict([(category, 0) for category in all_categories])

    current_count = 0
    for input_index in tqdm(range(len(input_data))):
        current_index = input_index
        input_prompt = input_data[current_index]
        current_count += 1

        failed = True
        retry_num = 0
        error = None
        while failed and retry_num < 10:
            try:
                is_toxic, moderation_response = moderator.is_toxic(input_prompt)
                category = check_category(moderation_response)
                failed = False
            except Exception as e:
                error = e.message
                retry_num += 1
        if failed:
            failed_df = pd.DataFrame({column:[input_prompt], "error_message": [error]})
            failed_df.to_csv(output_file_failed, index=False, header=not os.path.isfile(output_file_failed), mode='a')
            time.sleep(10)
            continue

        overall_stats[category] += 1
        if current_count % 1000 == 0:
            print(overall_stats)
        if column != 'prompt':
            df = pd.DataFrame({'prompt':[input_original_prompts[current_index]], column:[input_prompt], "category": [category], "is_toxic": [is_toxic], "moderation_result".format(moderator_model): moderation_response})
        else:
            df = pd.DataFrame({column:[input_prompt], "category": [category], "is_toxic": [is_toxic], "moderation_result".format(moderator_model): moderation_response})
        if merge_result:
            df.to_csv(output_file_merged, index=False, header=not os.path.isfile(output_file_merged), mode='a')
        else:
            if is_toxic:
                df.to_csv(output_file_toxic, index=False, header=not os.path.isfile(output_file_toxic), mode='a')
            else:
                df.to_csv(output_file_non_toxic, index=False, header=not os.path.isfile(output_file_non_toxic), mode='a')


if __name__ == "__main__":
    fire.Fire(main)

