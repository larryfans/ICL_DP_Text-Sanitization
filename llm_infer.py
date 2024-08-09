# import evaluate
import pandas as pd
import json
import models
from fastchat.model import get_conversation_template
# from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer
import re
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version_folder", type=str, default='./privatized_dataset/glove.42B.300d/conservative/')
    parser.add_argument("--version", type=str, default='eps_1.0_top_20_s1_save_stop_words_True')
    parser.add_argument("--is_map", type=bool, default=False, required=False)
    parser.add_argument("--shot",type=str,choices=['0', '2'])
    parser.add_argument("--save_folder",type=str, default='./Eval_Final/')
    return parser


Summarize_Prompt_Tamplete_2_shot = """Context 1: {dialogue1}
Response 1: {summary1}
Context 2: {dialogue2}
Response 2: {summary2}
The above is an example of a summary sentence, please help me summarise the following sentence.
Context:{private_dialogue}
Response: """


Summarize_Prompt_Tamplete_0_shot = """Please help me summarise the following sentence.
Context:{private_dialogue}
Response: """

def create_prompt(prompt,model_name):

    conv_template = get_conversation_template(
        model_name
    )
    conv_template.append_message(conv_template.roles[0], prompt)
    conv_template.append_message(conv_template.roles[1], None)
    if 'gpt' in model_name:
        full_prompt = conv_template.to_openai_api_messages()
    else:
        full_prompt = conv_template.get_prompt()
    # Clear the conv template
    # self.conv_template.messages = []
    return full_prompt

def replace_words(text, word_map):
    def replace(match):
        word = match.group(0)
        return word_map.get(word, word)
    
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in word_map.keys()) + r')\b')
    
    return pattern.sub(replace, text)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    print(args)
    # data_folder_path = '/root/fjl/research_works/dp_llm_work/CusText/CusText/dolly/privatized_dataset/glove.42B.300d/conservative/eps_1.0_top_20_s1_save_stop_words_True'

    with open(os.path.join(args.version_folder,args.version, 'train.json'), 'r', encoding='utf-8') as file:
            train_data = json.load(file)

    with open(os.path.join(args.version_folder,args.version, 'test.json'), 'r', encoding='utf-8') as file:
            test_data = json.load(file)

    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct' 
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


    llm = models.OpenAILLM(model_path='gpt-3.5-turbo')
    
    
    i = 1
    answer_data = []

    print('len: ', len(test_data))

    for item in  range(len(test_data)):
        print("------", i, '/', len(test_data), "------")
        # if i > 10:
        #     break
        if args.shot == '2':
            output = llm.generate(prompt=llm.create_conv_prompt(Summarize_Prompt_Tamplete_2_shot.format(
                dialogue1=train_data[item*2]['private_context'],
                summary1=train_data[item*2]['private_response'],
                dialogue2=train_data[item*2+1]['private_context'],
                summary2=train_data[item*2+1]['private_response'],
                private_dialogue=test_data[item]['private_context'])), 
                temperature=0.1, max_tokens=60)
        elif args.shot == '0':
            output = llm.generate(prompt=llm.create_conv_prompt(Summarize_Prompt_Tamplete_0_shot.format(
                private_dialogue=test_data[item]['context'])), 
                temperature=0.1, max_tokens=100)
            
        print("llm: ", output)
        # if need map
        reversed_map = {}
        if args.is_map == True:
            reversed_map = {v: k for k, v in test_data[item]['private_word_map'].items()}
        
        input_ids = tokenizer.encode(output, return_tensors='pt')
        # remap_text = []
        remap_token = []
        for index in range(input_ids.size(1)):
            token_str = tokenizer.decode([input_ids[0, index]])
            if token_str == tokenizer.bos_token:
                continue
            if int(input_ids[0, index]) in reversed_map:
                remap_token.append(int(reversed_map[int(input_ids[0, index])]))
                # print("map--")
            else:
                remap_token.append(int(input_ids[0, index]))
        remap_sentence = tokenizer.decode(remap_token)
        #     token_str = tokenizer.decode([input_ids[0, index]]).strip()
        #     if token_str in reversed_map:
        #         remap_text.append(reversed_map[token_str])
        #         print('-map-')
        #     else:
        #         remap_text.append(token_str)
        # remap_sentence = " ".join(remap_text)
        # new_text = replace_words(output, reversed_map)
        print("map answer: ", remap_sentence)
        
        print("label: ", test_data[item]['response'])

        answer_data.append({
            "response": test_data[item]['response'],
            "llm_response": output,
            "noised_remap_llm_response": remap_sentence,
            
        })
        
        i += 1
        
    import os
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    
    # with open(os.path.join(args.save_folder, args.shot+"_shot_raw"+".json"), 'w', encoding='utf-8') as f:
    with open(os.path.join(args.save_folder, args.shot+"_shot_"+args.version+".json"), 'w', encoding='utf-8') as f:
        json.dump(answer_data, f, ensure_ascii=False, indent=4)

