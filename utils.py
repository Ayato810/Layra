import re

from NovelAILLMWrapper.GenerationSettings import GenerationSettings
from novelai_api.Preset import Preset, Model
from novelai_api.BanList import BanList


def get_final_prompt(prompt, user, char):
    replacements = {
        "{{user}}": user,
        "{{char}}": char
    }
    for placeholder, replacement in replacements.items():
        pattern = re.escape(placeholder)
        prompt = re.sub(pattern, replacement, prompt)
    return prompt


def load_generation_settings(config_data):
    preset = Preset.from_official(Model.Kayra, "ProWriter")
    preset.min_length = 1
    preset.max_length = 150
    bad_words = BanList()
    for item in config_data["bad_words"]:
        bad_words.add(item)
    user = config_data["user"]
    char = config_data["char"]
    stop_sequences = ["\n", f"\n{user}", f"\n{char}"]
    generation_settings = GenerationSettings(preset=preset, stop_sequences=stop_sequences,
                                             repetition_penalty_whitelist=config_data["repetition_penalty_whitelist"],
                                             bad_words=bad_words)
    return generation_settings
