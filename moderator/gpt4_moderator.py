from config import moderation_template
class GPT4Moderator:
    def __init__(self, client):
        assert(client != None)
        self.client = client

    def is_toxic(self, prompt):
        generate_prompt_response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": moderation_template.format(prompt),
                }
            ],
            model='gpt-4-turbo-2024-04-09',
            temperature=0.0,
            max_tokens=1000
        )
        response = generate_prompt_response.choices[0].message.content

        if 'none of the above' in response.lower():
            return False, response
        else:
            return True, response
