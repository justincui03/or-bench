from mixtral_toxic_prompt_generator import MixtralToxicPromptGenerator
from mistralai.client import MistralClient
import fire
import os

# 
def main(category, output_path=None, num_toxic_prompts=2000):

    assert(category is not None)
    if output_path is None:
        output_path = './{}_prompts.csv'.format(category)
    mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
    generator = MixtralToxicPromptGenerator(mistral_client)
    generator.generate_through_loop(category=category, num_prompts=num_toxic_prompts, step_size=20, enable_tqdm=True, output_file_path=output_path)

if __name__ == '__main__':
    fire.Fire(main)







