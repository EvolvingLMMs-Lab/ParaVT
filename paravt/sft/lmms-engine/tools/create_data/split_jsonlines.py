import argparse

import jsonlines


def calculate_chunked_size(total_size, chunk_size):
    return (total_size + chunk_size - 1) // chunk_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--num-split", "-s", type=int, required=True)
    args = parser.parse_args()

    print("Loading data")
    with jsonlines.open(args.input) as reader:
        data = list(reader)
    print(f"Loaded {len(data)} data")

    chunked_size = calculate_chunked_size(len(data), args.num_split)
    file_name = args.input.split(".jsonl")[0]
    for i in range(args.num_split):
        start = i * chunked_size
        end = (i + 1) * chunked_size
        if end > len(data):
            end = len(data)
        with jsonlines.open(f"{file_name}_chunked_{i}.jsonl", "w") as writer:
            writer.write_all(data[start:end])
        print(f"Saved {file_name}_chunked_{i}.jsonl with {end - start} data")
