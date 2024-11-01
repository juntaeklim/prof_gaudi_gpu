from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch
import time

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
        
    model = args.model
    batch_size = args.batch_size
    input_length = args.input_length
    output_length=  args.output_length
    warmup = args.warmup
    n_iterations = args.n_iterations
    breakdown = args.breakdown
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        assert False
    
    prompt = "Here is my prompt: Vertically Integrated Architecture (VIA) research group is affiliated with the School of Electrical Engineering, Department of Semiconductor System Engineering, Graduate School of Artificial Intelligence (AI), and Graduate School of AI Semiconductor at KAIST, South Korea. We conduct research in the domain of computer architecture with a vertically integrated approach. By co-optimizing VLSI technology, computer system architecture, and application & algorithms, our mission is to build a high-performance computing platform for future \"intelligent\" systems that are programmable, robust, reliable, secure, and energy-efficient. (Note) For students interested in undergraduate research internships or those who are applying to our research group for graduate studies, please send me an email with your latest CV and transcript. I'm a Tenured Associate Professor at the School of Electrical Engineering, jointly affiliated with Department of Semiconductor System Engineering,  Graduate School of Artificial Intelligence (AI), and Graduate School of  AI Semiconductor at KAIST. I am was a Senior Research Scientist working at the architecture research group at NVIDIA. I had an opportunity to work on a number of exciting projects at NVIDIA that span several areas in computing, which include ASIC designs, computer system architecture, runtime systems, and application & workload characterization with an emphasis on deep neural networks (DNNs). Initially at NVIDIA, I worked on developing microarchitectural support for high-performance GPU cache replacement policies. More recently, I have been working in the domain of deep learning, trying to come up with neat architectural enhancements to the GPU hardware/software stack so that NVIDIA maintains its leadership in the areas of machine learning. For instance, I led the research and development of the virtualized DNN runtime system, a high-performance GPU memory virtualization solution for DNN training. I was also the technical lead on the architecture design, implementation, and evaluation of the sparse CNN accelerator, an ASIC developed by NVIDIA Research aiming towards achieving high energy-efficiency for DNN inference. In the past, I earned my Ph.D. degree from the University of Texas at Austin in 2014, under the guidance of professor Mattan Erez. I received my M.S. and B.E. degree from KAIST (Korea Advanced Institute of Science and Technology) and Sogang University, in 2009 and 2007, respectively."
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype)
    model = model.eval().to("cuda")
    
    # Adjust the input length
    tmp_input_tokens = tokenizer.encode(prompt)
    assert len(tmp_input_tokens) >= input_length
    final_prompt = tokenizer.decode(tmp_input_tokens[:input_length], skip_special_tokens=True)
    print("Batch size: %d, input: %s" %(batch_size, final_prompt))
    
    # Make batched prompt
    batched_prompt = []
    for _ in range(batch_size):
        batched_prompt.append(final_prompt)
    
    if breakdown:
        breakdown_times = []
    else:
        breakdown_times = None
        
    iteration_times = []
    for iter in range(warmup + n_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
            
        if breakdown:
            timestamps = []
        else:
            timestamps = None
        
        record_timestamp(timestamps)
        
        input_tokens = tokenizer.batch_encode_plus(batched_prompt, return_tensors="pt", padding=False)
        assert len(input_tokens["input_ids"]) == batch_size
        assert len(input_tokens["input_ids"][0]) == input_length
        
        record_timestamp(timestamps)
        
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to("cuda")
                
        output_tokens = model.generate(
                    **input_tokens,
                    do_sample=False,
                    max_length=output_length + input_length,
                    eos_token_id=None,
                ).cpu()

        record_timestamp(timestamps)

        output_sentences = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        
        record_timestamp(timestamps)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        iteration_times.append(end - start)
            
        if breakdown:
            breakdown_times += [timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)] 
    
    duration = sum(iteration_times[-n_iterations:])/n_iterations
    total_new_tokens_generated = batch_size * output_length
    throughput = total_new_tokens_generated / duration

    print()
    print("Results:")
    print(output_sentences[0])
    print()
    print("Input length: %d / output length: %d" %(len(input_tokens[0]), len(output_tokens[0])))
    print()
    print("Iteration times")
    print(iteration_times)
    print()
    print("Latency (msec), Throughput (token/sec)")
    print("%f, %f" %(duration * 1000, throughput))
    if breakdown:
        n_breakdown = 3    
        breakdown_times_tensor = torch.tensor(breakdown_times[-(n_breakdown*n_iterations):]).view(n_iterations, n_breakdown)
        final_times = torch.sum(breakdown_times_tensor, dim=1)
        final_breakdown = torch.mean(breakdown_times_tensor, dim=0)
        print(final_times * 1000)
        print(final_breakdown * 1000)
        print()
    
    for i in output_tokens:
        assert len(i) == output_length + input_length
    
    
    
if __name__ == "__main__":
    run()