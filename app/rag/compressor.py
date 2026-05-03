from typing import List

class ContextCompressor:
    def __init__(self):
        try:
            from llmlingua import PromptCompressor
            # Using LLMLingua-2 small model for faster CPU execution
            self.compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map="cpu"
            )
        except Exception as e:
            print(f"Failed to load compressor: {e}")
            self.compressor = None

    def compress(self, query: str, contexts: List[str]) -> List[str]:
        if not self.compressor or not contexts:
            return contexts
            
        context_str = "\n\n".join(contexts)
        try:
            results = self.compressor.compress_prompt_llmlingua2(
                context=context_str,
                chunk_end_tokens=[".", "\n"],
                target_token=800,  # compress down to 800 tokens
                rate=0.5
            )
            return [results["compressed_prompt"]]
        except Exception as e:
            print(f"Compression error: {e}")
            return contexts
