import random
import json
import os
from typing import List, Tuple

class PromptBuilder:
    def __init__(self, train_data_path: str, mode: str = 'zero-shot', shot_num: int = 0):
        self.mode = mode
        self.shot_num = shot_num

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.normpath(os.path.join(script_dir, train_data_path))

        with open(data_path, 'r', encoding='utf-8') as f:
            self.examples = [json.loads(line.strip()) for line in f]

    def build_prompt(self, input_question: str) -> str:
        instruction = "Translate the following natural language question into an SQL query."
        prompt = instruction + "\n\n"
        if self.mode == 'zero-shot':
            prompt += f"Question: {input_question}\nSQL:"
        else:
            samples = random.sample(self.examples, min(self.shot_num, len(self.examples)))
            for sample in samples:
                prompt += f"Question: {sample['text']}\nSQL: {sample['sql']}\n\n"
            prompt += f"Question: {input_question}\nSQL:"
        return prompt
