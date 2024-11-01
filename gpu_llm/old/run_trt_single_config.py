from transformers import AutoTokenizer
import argparse
import torch
import time
import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

def record_timestamp(timestamps):
    if timestamps is not None:
        torch.cuda.synchronize()
        timestamps.append(time.perf_counter())
def run():
    parser = argparse.ArgumentParser(description="OPT inference GPU test")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-length", type=int, default=10)
    parser.add_argument("--output-length", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--breakdown", action="store_true", default=False)
    parser.add_argument("--dtype", choices=["fp16", "fp32", "bf16"])
    args = parser.parse_args()
    batch_size = args.batch_size
    input_length = args.input_length
    output_length = args.output_length
    warmup = args.warmup
    n_iterations = args.n_iterations
    breakdown = args.breakdown
    if True:
        prompt = "Here is my prompt: Vertically Integrated Architecture (VIA) research group is affiliated with the School of Electrical Engineering, Department of Semiconductor System Engineering, Graduate School of Artificial Intelligence (AI), and Graduate School of AI Semiconductor at KAIST, South Korea. We conduct research in the domain of computer architecture with a vertically integrated approach. By co-optimizing VLSI technology, computer system architecture, and application & algorithms, our mission is to build a high-performance computing platform for future \"intelligent\" systems that are programmable, robust, reliable, secure, and energy-efficient. (Note) For students interested in undergraduate research internships or those who are applying to our research group for graduate studies, please send me an email with your latest CV and transcript. I'm a Tenured Associate Professor at the School of Electrical Engineering, jointly affiliated with Department of Semiconductor System Engineering,  Graduate School of Artificial Intelligence (AI), and Graduate School of  AI Semiconductor at KAIST. I am was a Senior Research Scientist working at the architecture research group at NVIDIA. I had an opportunity to work on a number of exciting projects at NVIDIA that span several areas in computing, which include ASIC designs, computer system architecture, runtime systems, and application & workload characterization with an emphasis on deep neural networks (DNNs). Initially at NVIDIA, I worked on developing microarchitectural support for high-performance GPU cache replacement policies. More recently, I have been working in the domain of deep learning, trying to come up with neat architectural enhancements to the GPU hardware/software stack so that NVIDIA maintains its leadership in the areas of machine learning. For instance, I led the research and development of the virtualized DNN runtime system, a high-performance GPU memory virtualization solution for DNN training. I was also the technical lead on the architecture design, implementation, and evaluation of the sparse CNN accelerator, an ASIC developed by NVIDIA Research aiming towards achieving high energy-efficiency for DNN inference. In the past, I earned my Ph.D. degree from the University of Texas at Austin in 2014, under the guidance of professor Mattan Erez. I received my M.S. and B.E. degree from KAIST (Korea Advanced Institute of Science and Technology) and Sogang University, in 2009 and 2007, respectively. On the 5th, KAIST announced that Minsoo Rhu, Professor in the Department of Electrical Engineering, has been appointed as Program Co-Chair for the IEEE/ACM International Symposium on Microarchitecture (MICRO) scheduled to be held next year. This marks the first time in MICRO’s 57-year history that a faculty member from an Asian university has been selected as Program Chair. Celebrating its 57th edition this year, MICRO is the oldest and most prestigious international conference in the field of computer architecture. Alongside ISCA and HPCA, it is regarded as one of the top three international conferences in computer architecture. Scholars and industry professionals from around the world participate in MICRO, with fewer than 20% of submitted papers being selected for final presentation. Professor Rhu was appointed Program Chair of the 58th MICRO conference, set to be held next year, in recognition of his contributions to the field of computer architecture. He will serve as Program Co-Chair alongside Professor Radu Teodorescu of Ohio State University, overseeing the selection of around 300 expert members of the Program Committee and supervising the review of over 500 submitted papers. Professor Rhu is recognized as a next-generation leader in the fields of intelligent semiconductors and computer systems for artificial intelligence (AI). His expertise is reflected in his induction into the Hall of Fame of major conferences, including HPCA in 2021, MICRO in 2022, and ISCA this year. Professor Rhu completed his undergraduate studies in electronic engineering at Sogang University, obtained his master’s degree in electrical engineering from KAIST, and earned his Ph.D. in computer science from the University of Texas at Austin. From 2014 to 2017, he worked at NVIDIA Research, and since 2018, he has been a professor at KAIST. He also served as a visiting researcher at Meta AI from 2022 to last year. His research has been widely recognized by academia, receiving the Best Paper Award at HPCA this year, the Google Faculty Research Award last year, and the Facebook Faculty Research Award in 2020. Last year, he was also inducted as a member of Y-KAST, an elite group of young scientists under 43 recognized for their outstanding contributions to science by the Korean Academy of Science and Technology."
        
    
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Prepare the input tokens
    tmp_input_tokens = tokenizer.encode(prompt)
    assert len(tmp_input_tokens) >= input_length
    final_prompt = tokenizer.decode(tmp_input_tokens[:input_length], skip_special_tokens=True)
    print("Batch size: %d, input: %s" % (batch_size, final_prompt))
    # Create batched prompt
    batched_prompt = [final_prompt for _ in range(batch_size)]
    if breakdown:
        breakdown_times = []
    else:
        breakdown_times = None
    iteration_times = []
    # Initialize the model runner for TensorRT
    runner_cls = ModelRunner
    # runner_cls = ModelRunnerCpp
    # runner_cls = ModelRunnerCpp if args.use_cpp_session else ModelRunner
    if args.model == "facebook/opt-125m":
        engine_dir = "opt/125M/trt_engines/fp16/1-gpu/"
    elif args.model == "facebook/opt-1.3b":
        engine_dir = "opt/1.3B/trt_engines/fp16/1-gpu/"
    elif args.model == "facebook/opt-6.7b" and args.dtype == "bf16":
        engine_dir = "opt/6.7B/trt_engines/bf16/1-gpu/"
    elif args.model == "meta-llama/Llama-3.1-8B-Instruct" and args.dtype == "fp16":
        engine_dir = "TensorRT-LLM/examples/llama/tmp/llama/8B/trt_engines/fp16/1-gpu/"
    elif args.model == "meta-llama/Llama-3.1-8B-Instruct" and args.dtype == "bf16":
        engine_dir = "TensorRT-LLM/examples/llama/tmp/llama/8B/trt_engines/bf16/1-gpu/"
    else:
        assert False
        
    
    runner_kwargs = {
        'engine_dir': engine_dir,
        'rank': tensorrt_llm.mpi_rank(),
        'max_output_len': output_length,
        # Add other parameters as necessary
    }
    runner = runner_cls.from_dir(**runner_kwargs)
    
    for iter in range(warmup + n_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if breakdown:
            timestamps = []
        else:
            timestamps = None
        record_timestamp(timestamps)
        # Prepare the input tokens for TensorRT
        input_tokens = tokenizer.batch_encode_plus(batched_prompt, return_tensors="pt", padding=False)
        assert len(input_tokens["input_ids"]) == batch_size
        assert len(input_tokens["input_ids"][0]) == input_length
        record_timestamp(timestamps)
        for key in input_tokens:
            if torch.is_tensor(input_tokens[key]):
                input_tokens[key] = input_tokens[key].to("cuda")
        # Run inference using the TensorRT model runner
        output_tokens = runner.generate(
            batch_input_ids=input_tokens["input_ids"],
            max_new_tokens=output_length,
            end_id=-1,  # Set as per your requirement
            temperature=1.0,  # Modify as needed
            pad_id=-1,
            do_sample=False,
            # max_num_tokens=batch_size * input_length
            # Add other parameters for the generation
        ).cpu()
        record_timestamp(timestamps)
        output_sentences = tokenizer.batch_decode(output_tokens.squeeze(), skip_special_tokens=True)
        record_timestamp(timestamps)
        torch.cuda.synchronize()
        end = time.perf_counter()
        iteration_times.append(end - start)
        if breakdown:
            breakdown_times += [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    duration = sum(iteration_times[-n_iterations:]) / n_iterations
    total_new_tokens_generated = batch_size * output_length
    throughput = total_new_tokens_generated / duration
    print()
    print("Results:")
    print(output_sentences[0])
    print()
    print("Input length: %d / output length: %d" % (len(input_tokens["input_ids"][0]), len(output_tokens[0][0])))
    print(f"output tokens.size(): {output_tokens.size()}")
    print()
    print("Iteration times")
    print(iteration_times)
    print()
    print("Latency (msec), Throughput (token/sec)")
    print("%f, %f" % (duration * 1000, throughput))
    if breakdown:
        n_breakdown = 3
        breakdown_times_tensor = torch.tensor(breakdown_times[-(n_breakdown * n_iterations):]).view(n_iterations, n_breakdown)
        final_times = torch.sum(breakdown_times_tensor, dim=1)
        final_breakdown = torch.mean(breakdown_times_tensor, dim=0)
        print(final_times * 1000)
        print(final_breakdown * 1000)
        print()
    for i in output_tokens:
        assert i.shape[1] == output_length + input_length
if __name__ == "__main__":
    run()
