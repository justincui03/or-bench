import pandas as pd
import fire
from mistralai.client import MistralClient
from mixtral_overalign_rewritter import MixtralOveralignRewritter
from tqdm import tqdm
import os


def main(toxic_seed_path):
    """
    This file will read from toxic_seed_path, rewrite them into seemingly toxic prompts.
    """

    df = pd.read_csv(toxic_seed_path)
    output_path = './' + toxic_seed_path.split("/")[-1][:-4] + '_rewrite.csv'

    mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
    prompt_rewritter = MixtralOveralignRewritter(mistral_client)

    already_generated = set()
    for index in tqdm(range(len(df))):
        toxic_prompt = df.iloc[index]['prompt']
        toxic_category = df.iloc[index]['category']

        rewritten_prompts, _ = prompt_rewritter.rewrite(toxic_prompt)
        rewritten_prompts = [x for x in rewritten_prompts if x not in already_generated]
        already_generated.update(rewritten_prompts)
        output_df = pd.DataFrame({'prompt': rewritten_prompts, "toxic_seed": [toxic_prompt] * len(rewritten_prompts), "category":[toxic_category] * len(rewritten_prompts)})
        output_df.to_csv(output_path, mode='a', index=False, header=not os.path.isfile(output_path))

if __name__ == '__main__':
    fire.Fire(main)

