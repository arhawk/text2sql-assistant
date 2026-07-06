import os
import json
import random
import re

class PromptBuilder:
    def __init__(self, train_data_path: str, mode: str = 'zero-shot', shot_num: int = 0):
        self.mode = mode
        self.shot_num = shot_num

        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.normpath(os.path.join(script_dir, train_data_path))

        with open(data_path, 'r', encoding='utf-8-sig') as f:
            self.examples = [json.loads(line.strip()) for line in f if line.strip()]

    def sanitize_text(self, text):
        text = text.replace('\"', '"')
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)  # 移除控制字符
        text = re.sub(r'\s+', ' ', text).strip()  # 合并多余空格
        return text

    def build_prompt(self, input_question: str) -> str:
        instruction = "Translate the following natural language question into an SQL query."
        prompt = instruction + "\n\n"
        if self.mode == 'zero-shot':
            prompt += f"Question: {self.sanitize_text(input_question)}\nSQL:"
        else:
            samples = random.sample(self.examples, min(self.shot_num, len(self.examples)))
            for i, sample in enumerate(samples):
                q = self.sanitize_text(sample['text'])
                a = self.sanitize_text(sample['sql'])
                prompt += f"# Example {i+1}\nQuestion: {q}\nSQL: {a}\n\n"
            prompt += f"# Now answer this one\nQuestion: {self.sanitize_text(input_question)}\nSQL:"
        print("\n🧪 Prompt Preview:", repr(prompt[:500]))
        return prompt
