from config import moderation_template
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class GeminiModerator:
    def __init__(self, gemini_client):
        assert(gemini_client != None)
        self.gemini_client = gemini_client

    def is_toxic(self, prompt):
        gemini_response = self.gemini_client.generate_content(moderation_template.format(prompt),
                                                     safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT:HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH:HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:HarmBlockThreshold.BLOCK_NONE,
    }  )
        response = ''
        if (len(gemini_response.candidates) == 0):
            response = str(gemini_response)
        elif (len(gemini_response.candidates) == 1):
            try:
                response = gemini_response.text
            except ValueError:
                print(gemini_response.prompt_feedback)
                print(gemini_response.candidates[0].finish_reason)
                print(gemini_response.candidates[0].safety_ratings)
                response = str(gemini_response)
        else:
            response = str(gemini_response)
        if 'none of the above' in response.lower():
            return False, response
        else:
            return True, response
    