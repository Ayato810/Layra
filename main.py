import json
import dotenv
from NovelAILLMWrapper.NovelAILLM import NovelAILLM
import utils


def layra():
    dotenv.load_dotenv()
    with open('config.json', 'r', encoding='utf-8') as config_file:
        config_data = json.load(config_file)

    attg = config_data["attg"]
    preamble = config_data["preamble"]
    user_card = config_data["user_card"]
    char_card = config_data["char_card"]
    style = config_data["style"]
    speech_examples = config_data["speech_examples"]
    first_message = """\n"""

    prompt = attg + preamble + user_card + char_card + style + speech_examples + "***" + first_message
    final_prompt = utils.get_final_prompt(prompt, config_data["user"], config_data["char"])

    generation_settings = utils.load_generation_settings(config_data)
    llm = NovelAILLM(generation_settings=generation_settings)
    gen = llm(final_prompt)
    print("output: " + gen)


if __name__ == '__main__':
    layra()
