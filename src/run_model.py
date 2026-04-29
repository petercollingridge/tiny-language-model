import os
from utils import parse_model_data


def add_lists(list1, list2):
    return [a + b for a, b in zip(list1, list2)]


def run_model(folder, filename="model_output.txt"):
    filepath = os.path.join(os.path.dirname(__file__), folder, filename)
    model_data = parse_model_data(filepath)

    token_embeddings = model_data['weights'][0]
    pos_embeddings = model_data['weights'][1]
    key = model_data['weights'][2]
    query = model_data['weights'][3]
    value = model_data['weights'][4]
    lm_head = model_data['weights'][5]
    bias = model_data['weights'][6]

    tokens = model_data['tokens']
    print("tokens", tokens)

    prompt = '<BR>'
    indices = tokens.index(prompt)
    print("prompt_tokens", indices)

    token_embedding = token_embeddings[indices]
    print("token_embedding", token_embedding)

    pos_embedding = pos_embeddings[indices]
    print("pos_embedding", pos_embedding)

    print("token_embedding + pos_embedding", add_lists(token_embedding, pos_embedding))

    


if __name__ == "__main__":
    run_model("example4", "model_output_1_good.txt")
