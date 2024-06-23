from config import moderation_template
class LlamaModerator:
    def __init__(self, client):
        assert(client != None)
        self.client = client

    def is_toxic(self, prompt):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                "role": "user",
                "content": moderation_template.format(prompt),
                }
            ],
            model="meta-llama/llama-3-70b-chat-hf",
            max_tokens=1000,
            temperature=0.0,
        )

        response = chat_completion.choices[0].message.content

        if 'none of the above' in response.lower():
            return False, response
        else:
            return True, response
    