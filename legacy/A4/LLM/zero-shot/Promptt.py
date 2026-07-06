import random
import json
from typing import List, Tuple

class PromptBuilder:
    def __init__(self, train_data_path: str, mode: str = 'zero-shot', shot_num: int = 0):
        """
        :param train_data_path: path to generation_train.jsonl
        :param mode: one of ['zero-shot', 'few-shot', 'many-shot']
        :param shot_num: number of examples to include (ignored for zero-shot)
        """
        self.mode = mode
        self.shot_num = shot_num

        with open(train_data_path, 'r', encoding='utf-8') as f:
            self.examples = [json.loads(line.strip()) for line in f]

    def build_prompt(self, input_question: str) -> str:
        """Return prompt string ready for LLM input"""
        instruction = "Translate the following natural language question into an SQL query."
        prompt = instruction + "\n\n"

        if self.mode == 'zero-shot':
            prompt += f"Question: {input_question}\nSQL:"
        else:
            samples = random.sample(self.examples, min(self.shot_num, len(self.examples)))
            for ex in samples:
                prompt += f"Question: {ex['text']}\nSQL: {ex['sql']}\n\n"
            prompt += f"Question: {input_question}\nSQL:"

        return prompt


# Example usage
if __name__ == '__main__':
    builder = PromptBuilder('../../processed_data/question_split/generation_train.jsonl', mode='zero-shot')
    q = "what are the flights from dallas to boston on july 5"
    prompt = builder.build_prompt(q)
    print(prompt)
