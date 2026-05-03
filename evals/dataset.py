import pandas as pd

# We use a small subset of 5 questions for rapid CI/CD baseline testing
# In a real setup, this would load 100 questions from a JSON/CSV.
eval_data = [
    {
        "question": "What is the role of attention mechanism in transformers?",
        "ground_truth": "The attention mechanism allows the model to focus on relevant parts of the input sequence when generating an output sequence, calculating a weighted sum of values based on query and key compatibility.",
    },
    {
        "question": "How does Reinforcement Learning from Human Feedback (RLHF) improve language models?",
        "ground_truth": "RLHF fine-tunes language models using human preferences as a reward signal, helping to align the model's outputs with human values and intent, reducing harmful or incorrect generations.",
    },
    {
        "question": "What is the difference between extractive and abstractive summarization?",
        "ground_truth": "Extractive summarization selects and concatenates exact sentences from the original text, whereas abstractive summarization generates new sentences to capture the core meaning.",
    },
    {
        "question": "Why is layer normalization used in deep neural networks?",
        "ground_truth": "Layer normalization stabilizes the training process and reduces training time by normalizing the inputs across the features for each training example.",
    },
    {
        "question": "What is the vanishing gradient problem?",
        "ground_truth": "The vanishing gradient problem occurs when gradients used to update network weights become extremely small during backpropagation, effectively preventing early layers from learning.",
    }
]

def load_eval_dataset():
    return pd.DataFrame(eval_data)
