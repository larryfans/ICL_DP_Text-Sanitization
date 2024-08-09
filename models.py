from fastchat.model import get_conversation_template
import openai

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


# set openai key
openai.api_key = '/api_key/'

openai.base_url = "/base_url/"
openai.default_headers = {"x-foo": "true"}

class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 temperature=0.01,
                 max_tokens=150,
                 ):
        super().__init__()
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        # self.client = openai.OpenAI()
        # self.client.api_key = os.environ["OPENAI_API_KEY"]

    # def set_system_message(self, conv_temp):
    #     if self.system_message is not None:
    #         conv_temp.set_system_message(self.system_message)
    #DEFAULT VALUES
    def generate(self, prompt,max_tokens=50,temperature=1.0,top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1):
        try:
            
            tip = False
            while tip == False:
                try:
                    response = openai.chat.completions.create(
                        model=self.model_path,
                        # messages=self.create_conv_prompt(prompt),
                        #case already handled in the calling functions
                        messages=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        n=n
                    )
                    tip = True
                except Exception as e:
                    print("Error with openai API, sleeping and again")
                    print(e)
                    # time.sleep(3)
            if n > 1:
                output = [choice.message.content for choice in response.choices]
            else:
                output = response.choices[0].message.content
            return output
        except Exception as e:
            # Check if the exception is a 400 Bad Request
            if hasattr(e, 'response') and e.response.status_code == 400:
                print("400 Bad Request error: ", e.response.text)
                # Handle the error as appropriate (e.g., return a default response or log the error)
                raise
            else:
                # Re-raise the exception if it's not a 400 Bad Request
                raise

    def generate_batch(self, prompts,max_tokens=100,temperature=1.0,top_p=1.0,frequency_penalty=0.0,presence_penalty=0.0,n=1):
        return [self.generate(prompt,max_tokens=max_tokens,temperature=temperature,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,n=n) for prompt in prompts]


    def create_conv_prompt(self,prompt,system_message=False, system_message_text="You are a good bot"):
        conv_template = get_conversation_template(
            "gpt-3.5-turbo"
        )
        if system_message:
            conv_template.set_system_message(system_message_text)
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        
        full_prompt = conv_template.to_openai_api_messages()
        return full_prompt


    def create_prompt(self,prompt):
        conv_template = get_conversation_template(
            "gpt-3.5-turbo"
        )
        conv_template.set_system_message("You are a helpful agent")
        # conv_template.set_system_message("Your objective is to determine if the user's request or question is harmful or not. " \
        #         + "Just answer \"harmful\" or \"not harmful\" to each user message. " \
        #         + "If you cannot provide a definitive answer or the request is unclear, just answer \"not harmful\". " \
        #         + "If the request is too short and incomplete, just answer \"not harmful\". " \
        #         + "Do not write anything else. Do not write anything else.")
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], None)
        
        full_prompt = conv_template.to_openai_api_messages()

        return full_prompt