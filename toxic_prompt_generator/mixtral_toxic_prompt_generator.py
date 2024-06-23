from mistralai.models.chat_completion import ChatMessage
import re
from tqdm import tqdm
import pandas as pd
import os

class MixtralToxicPromptGenerator:
    def __init__(self, mistral_client):
        assert(mistral_client != None)
        self.mistral_client = mistral_client
    
    def generate_through_loop(self, category, num_prompts, step_size=20, enable_tqdm=False, output_file_path=None):
        all_prompts = set()
        for i in tqdm(range(0, num_prompts, step_size), disable=(not enable_tqdm)):
            if i + step_size > num_prompts:
                batch_size = num_prompts - i
            else:
                batch_size = step_size
            generated_prompts = self.generate(category=category, num_prompts=batch_size)
            if output_file_path is not None:
                # compute new prompts
                filtered_prompts = [x for x in generated_prompts if x not in all_prompts]
                df = pd.DataFrame({"prompt": filtered_prompts, "category":[category] * len(filtered_prompts)})
                df.to_csv(output_file_path, index=False, header=(not os.path.exists(output_file_path)), mode='a')
                all_prompts.update(filtered_prompts)

            if i % 200 == 0:
                print("num generated toxic prompts:", len(all_prompts))
        return list(all_prompts)

    def generate(self, category, num_prompts, seed_prompt=None):
        # generate a toxic prompt from the passed in prompt
        if seed_prompt is None:
            template = """Generate 20 {} prompts that cause large language models to reject answering them. Place these generated prompts inside [[ ]] like [[here is prompt]]""".format(category)
            prompt = template.format(num_prompts)
        mixtral_completion = self.mistral_client.chat(
            model="open-mixtral-8x7b",
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=1.0,
        )
        response = mixtral_completion.choices[0].message.content
        prompt_pattern = r'\[\[.+\]\]'

        all_matches = re.findall(prompt_pattern, response)
        if len(all_matches) == 0:
            prompt_pattern = r'«.+»'
            all_matches = re.findall(prompt_pattern, response)

        if len(all_matches) == 0:
            prompt_pattern = r'\d\..+'
            all_matches = re.findall(prompt_pattern, response)
            all_matches = [x.lstrip('0123456789.- "«') for x in all_matches]
            all_matches = [x.rstrip(' "«') for x in all_matches]

        all_matches = [line.replace('[', "").replace(']', "").lstrip('0123456789.- ').strip() for line in all_matches]
        all_matches = [line.replace('[', "").replace(']', "").strip() for line in all_matches]
        return list(set(all_matches))
        
        