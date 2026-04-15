import os


def get_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def parse_data(text):
    lines = text.splitlines()
    tokens_line = lines[1]
    tokens = tokens_line.split(':')[1].strip().split('|')

    weights = []
    matrix = None
    for line in lines[2:]:
        if line.strip():
            if line.startswith("weights:"):
                if matrix is not None:
                    weights.append(matrix)
                matrix = []
            else:
                weight = [float(x) for x in line.strip().split(",")]
                matrix.append(weight)

    if matrix is not None:
        weights.append(matrix)

    return { 'tokens': tokens, 'weights': weights }

t = get_data(os.path.join("example1", "model_output.txt"))
p = parse_data(t)
print(p)
