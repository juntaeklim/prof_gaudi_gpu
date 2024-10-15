import os
import argparse

def read_log_file(file_path):
    """
    Reads the last line of a log file and returns the latency and throughput.
    If the last line starts with "torch.OutOfMemoryError", return None.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                assert False
                return None
            
            last_line = lines[-1].strip()
            if last_line.startswith("torch.OutOfMemoryError"):
                return None
            # Parse the last line which contains two numbers separated by a comma
            latency, throughput = map(float, last_line.split(","))
            return latency, throughput
    except FileNotFoundError:
        assert False
        return None


def main():
    parser = argparse.ArgumentParser(description="Parse log files and output latency and throughput")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'facebook/opt-1.3b')")
    parser.add_argument("--input-length", type=int, required=True, help="Input length (I)")
    parser.add_argument("--output-length", type=int, required=True, help="Output length (O)")
    args = parser.parse_args()

    # Model name conversion (replace '/' with '_')
    model_name = args.model.replace("/", "_")
    input_length = args.input_length
    output_length = args.output_length

    print(f"Used model: {args.model} / Input length: {input_length} / Output length: {output_length}")
    print("\n(Unit of latency = msec)")
    print("(Unit of throughput = token/sec)")
    print("\nbatch size, Latency, Throughput")

    # Check batch sizes (1, 2, 4, 8, ...)
    batch_size = 1
    while batch_size <= 1024:
        log_filename = f"./logs/{model_name}_batch_{batch_size}_in_{input_length}_out_{output_length}.txt"
        result = read_log_file(log_filename)

        if result is None:
            break

        latency, throughput = result
        print(f"{batch_size}, {latency}, {throughput}")

        # Increase batch size (multiply by 2)
        batch_size *= 2


if __name__ == "__main__":
    main()
