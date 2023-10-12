import re


def get_final_prompt(prompt, user, char):
    replacements = {
        "{{user}}": user,
        "{{char}}": char
    }

    final_prompt = ""
    for placeholder, replacement in replacements.items():
        pattern = re.escape(placeholder)
        final_prompt = re.sub(pattern, replacement, prompt)

    return final_prompt
