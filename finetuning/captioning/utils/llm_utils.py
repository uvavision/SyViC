from typing import List
import torch


class HuggingfaceHandler:
    MODEL_MAPPING = {
        "bloomz": "bigscience/bloomz-7b1",
        "flan-t5": "google/flan-t5-xxl",
    }

    def __init__(self, checkpoints: List[str] = ["bloomz", "flan-t5"]):
        from transformers import BloomTokenizerFast, BloomForCausalLM
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.checkpoints = checkpoints

        if "flan-t5" in checkpoints:
            checkpoint = HuggingfaceHandler.MODEL_MAPPING["flan-t5"]
            self.flan_t5_tokenizer = T5Tokenizer.from_pretrained(checkpoint)
            self.flan_t5_model = T5ForConditionalGeneration.from_pretrained(
                checkpoint,
                device_map="auto",
                torch_dtype=torch.float16,
                temperature=0.5,
            )
            self.flan_t5_model.eval()
        else:
            self.flan_t5_tokenizer, self.flan_t5_model = None, None

        if "bloomz" in checkpoints:
            checkpoint = HuggingfaceHandler.MODEL_MAPPING["bloomz"]
            self.bloomz_tokenizer = BloomTokenizerFast.from_pretrained(checkpoint)
            self.bloomz_model = BloomForCausalLM.from_pretrained(
                checkpoint,
                device_map="auto",
                torch_dtype=torch.float16,
                temperature=0.5,
            )
            self.bloomz_model.eval()
        else:
            self.bloomz_tokenizer, self.bloomz_model = None, None

    def handle_request(self, in_texts: str, checkpoint: str, **kwargs):
        if isinstance(in_texts, str):
            in_texts = [in_texts]

        if checkpoint == "bloomz":
            return self._handle_bloomz_request(in_texts, **kwargs)

        if checkpoint == "flan-t5":
            return self._handle_flan_t5_request(in_texts, **kwargs)

        return

    @torch.no_grad()
    def _handle_bloomz_request(self, in_texts, prefix=""):
        if not self.bloomz_model:
            return
        batch_tokens = self.bloomz_tokenizer.batch_encode_plus(
            in_texts, padding=True, return_tensors="pt"
        ).to("cuda")
        inputs, mask = batch_tokens["input_ids"], batch_tokens["attention_mask"]
        outputs = self.bloomz_model.generate(
            input_ids=inputs, attention_mask=mask, min_new_tokens=57, max_new_tokens=147
        )
        out_txt = self.bloomz_tokenizer.batch_decode(outputs)
        for i in range(len(out_txt)):
            out_txt[i] = out_txt[i].replace("</s>", "").replace("<pad>", "")
            out_txt[i] = out_txt[i].replace(in_texts[i], prefix).strip()
        return out_txt

    @torch.no_grad()
    def _handle_flan_t5_request(self, in_texts, prefix=""):
        if not self.flan_t5_model:
            return
        batch_tokens = self.flan_t5_tokenizer.batch_encode_plus(
            in_texts, padding=True, return_tensors="pt"
        ).to("cuda")
        inputs, mask = batch_tokens["input_ids"], batch_tokens["attention_mask"]
        outputs = self.flan_t5_model.generate(
            input_ids=inputs, attention_mask=mask, min_new_tokens=57, max_new_tokens=147
        )

        out_txt = self.flan_t5_tokenizer.batch_decode(outputs)
        for i in range(len(out_txt)):
            out_txt[i] = out_txt[i].replace("</s>", "").replace("<pad>", "").strip()
            out_txt[i] = f"{prefix} {out_txt[i]}"
        return out_txt


if __name__ == "__main__":
    checkpoints = ["bloomz", "flan-t5"]
    huggingface_handler = HuggingfaceHandler(checkpoints)
    print("Loaded Huggingface Models")

    in_texts = [
        "Going Beyond Nouns With Vision & Language Models Using Synthetic Data, we can see that",
        "In this scene, an apple falls on the head of a human.",
    ]
    prefix = "In this scene, we can see that"

    print("Bloomz output:")
    print(huggingface_handler.handle_request(in_texts, "bloomz", prefix=prefix))

    print("Flan-T5 output:")
    print(huggingface_handler.handle_request(in_texts, "flan-t5", prefix=prefix))
