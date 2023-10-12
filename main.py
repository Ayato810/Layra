import json
import dotenv
from NovelAILLMWrapper.NovelAILLM import NovelAILLM
from NovelAILLMWrapper.GenerationSettings import GenerationSettings
from novelai_api.BanList import BanList
from novelai_api.Preset import Model, Preset
from novelai_api.Tokenizer import Tokenizer
import utils


def layra():
    dotenv.load_dotenv()
    with open('config.json', 'r') as config_file:
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

    preset = Preset.from_official(Model.Kayra)
    preset.min_length = 1
    preset.max_length = 150

    preset.stop_sequences = [[85], [49308,
                                    49215,
                                    49285,
                                    11498,
                                    49287],
                             [49308,
                              49215,
                              46001,
                              49287]]

    stop_sequences = []
    for sequence in preset.stop_sequences:
        print(sequence)
        decoded_sequence = Tokenizer.decode(Model.Kayra, sequence)
        stop_sequences.append(decoded_sequence)
        print(Tokenizer.encode(Model.Kayra, decoded_sequence))

    print(stop_sequences)

    preset.repetition_penalty_whitelist = [[
        49256, 49264, 49231, 49230, 49287, 85, 49255, 49399,
        49262, 336, 333, 432, 363, 468, 492, 745,
        401, 426, 623, 794, 1096, 2919, 2072, 7379,
        1259, 2110, 620, 526, 487, 16562, 603, 805,
        761, 2681, 942, 8917, 653, 3513, 506, 5301,
        562, 5010, 614, 10942, 539, 2976, 462, 5189,
        567, 2032, 123, 124, 125, 126, 127, 128,
        129, 130, 131, 132, 588, 803, 1040, 49209,
        4, 5, 6, 7, 8, 9, 10, 11,
        12
    ]]

    repetition_penalty_whitelist = []

    for sequence in preset.repetition_penalty_whitelist:
        decoded_sequence = Tokenizer.decode(Model.Kayra, sequence)
        repetition_penalty_whitelist.append(decoded_sequence)

    print(repetition_penalty_whitelist)

    bad_words = BanList([23], [49209, 23], [23],
                        [49209, 23], [23], [49209, 23],
                        [23], [49209, 23], [23],
                        [49209, 23], [21], [49209, 21],
                        [21], [49209, 21], [21],
                        [49209, 21], [21], [49209, 21],
                        [21], [49209, 21], [3],
                        [49356], [1431], [31715],
                        [34387], [20765], [30702],
                        [10691], [49333], [1266],
                        [26523], [41471], [2936],
                        [85, 85], [49332], [7286],
                        [1115])

    bad_words_decoded = BanList()

    for sequence in bad_words:
        decoded_sequence = Tokenizer.decode(Model.Kayra, sequence)
        bad_words_decoded.add(decoded_sequence)

    print(bad_words_decoded)

    generation_settings = GenerationSettings(preset=preset, bad_words=bad_words_decoded,
                                             repetition_penalty_whitelist=repetition_penalty_whitelist,
                                             stop_sequences=stop_sequences)

    llm = NovelAILLM(generation_settings=generation_settings)
    gen = llm(final_prompt)
    print("output: " + gen)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    layra()
