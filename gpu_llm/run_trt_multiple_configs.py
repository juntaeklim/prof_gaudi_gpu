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

def run():
    parser = argparse.ArgumentParser(description="OPT inference GPU test")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--dtype", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--n-gpus", type=int, default=1)
    args = parser.parse_args()
    
    model = args.model
    warmup = args.warmup
    n_iterations = args.n_iterations
    dtype = args.dtype
    
    # Long prompt
    if True:
        prompt = "Here is my prompt: Vertically Integrated Architecture (VIA) research group is affiliated with the School of Electrical Engineering, Department of Semiconductor System Engineering, Graduate School of Artificial Intelligence (AI), and Graduate School of AI Semiconductor at KAIST, South Korea. We conduct research in the domain of computer architecture with a vertically integrated approach. By co-optimizing VLSI technology, computer system architecture, and application & algorithms, our mission is to build a high-performance computing platform for future \"intelligent\" systems that are programmable, robust, reliable, secure, and energy-efficient. (Note) For students interested in undergraduate research internships or those who are applying to our research group for graduate studies, please send me an email with your latest CV and transcript. I'm a Tenured Associate Professor at the School of Electrical Engineering, jointly affiliated with Department of Semiconductor System Engineering,  Graduate School of Artificial Intelligence (AI), and Graduate School of  AI Semiconductor at KAIST. I am was a Senior Research Scientist working at the architecture research group at NVIDIA. I had an opportunity to work on a number of exciting projects at NVIDIA that span several areas in computing, which include ASIC designs, computer system architecture, runtime systems, and application & workload characterization with an emphasis on deep neural networks (DNNs). Initially at NVIDIA, I worked on developing microarchitectural support for high-performance GPU cache replacement policies. More recently, I have been working in the domain of deep learning, trying to come up with neat architectural enhancements to the GPU hardware/software stack so that NVIDIA maintains its leadership in the areas of machine learning. For instance, I led the research and development of the virtualized DNN runtime system, a high-performance GPU memory virtualization solution for DNN training. I was also the technical lead on the architecture design, implementation, and evaluation of the sparse CNN accelerator, an ASIC developed by NVIDIA Research aiming towards achieving high energy-efficiency for DNN inference. In the past, I earned my Ph.D. degree from the University of Texas at Austin in 2014, under the guidance of professor Mattan Erez. I received my M.S. and B.E. degree from KAIST (Korea Advanced Institute of Science and Technology) and Sogang University, in 2009 and 2007, respectively. On the 5th, KAIST announced that Minsoo Rhu, Professor in the Department of Electrical Engineering, has been appointed as Program Co-Chair for the IEEE/ACM International Symposium on Microarchitecture (MICRO) scheduled to be held next year. This marks the first time in MICRO’s 57-year history that a faculty member from an Asian university has been selected as Program Chair. Celebrating its 57th edition this year, MICRO is the oldest and most prestigious international conference in the field of computer architecture. Alongside ISCA and HPCA, it is regarded as one of the top three international conferences in computer architecture. Scholars and industry professionals from around the world participate in MICRO, with fewer than 20% of submitted papers being selected for final presentation. Professor Rhu was appointed Program Chair of the 58th MICRO conference, set to be held next year, in recognition of his contributions to the field of computer architecture. He will serve as Program Co-Chair alongside Professor Radu Teodorescu of Ohio State University, overseeing the selection of around 300 expert members of the Program Committee and supervising the review of over 500 submitted papers. Professor Rhu is recognized as a next-generation leader in the fields of intelligent semiconductors and computer systems for artificial intelligence (AI). His expertise is reflected in his induction into the Hall of Fame of major conferences, including HPCA in 2021, MICRO in 2022, and ISCA this year. Professor Rhu completed his undergraduate studies in electronic engineering at Sogang University, obtained his master’s degree in electrical engineering from KAIST, and earned his Ph.D. in computer science from the University of Texas at Austin. From 2014 to 2017, he worked at NVIDIA Research, and since 2018, he has been a professor at KAIST. He also served as a visiting researcher at Meta AI from 2022 to last year. His research has been widely recognized by academia, receiving the Best Paper Award at HPCA this year, the Google Faculty Research Award last year, and the Facebook Faculty Research Award in 2020. Last year, he was also inducted as a member of Y-KAST, an elite group of young scientists under 43 recognized for their outstanding contributions to science by the Korean Academy of Science and Technology."
        
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Initialize the model runner for TensorRT
    runner_cls = ModelRunner
    # runner_cls = ModelRunnerCpp
    # runner_cls = ModelRunnerCpp if args.use_cpp_session else ModelRunner
    
    if args.model == "facebook/opt-6.7b" and args.n_gpus == 1:
        engine_dir = "opt/6.7B/trt_engines/bf16/1-gpu/"
    elif args.model == "facebook/opt-66b" and args.n_gpus in 2:
        engine_dir = "opt/66B/trt_engines/bf16/2-gpu/"
    elif args.model == "facebook/opt-66b" and args.n_gpus == 4:
        engine_dir = "opt/66B/trt_engines/bf16/4-gpu/"
    elif args.model == "facebook/opt-66b" and args.n_gpus == 8:
        engine_dir = "opt/66B/trt_engines/bf16/8-gpu/"
    elif args.model == "meta-llama/Llama-3.1-8B-Instruct" and args.n_gpus == 1:
        engine_dir = "llama/8B/trt_engines/bf16/1-gpu_no_maxnumtokens/"
    elif args.model == "meta-llama/Llama-3.1-70B-Instruct" and args.n_gpus == 2:
        engine_dir = "llama/70B/trt_engines/bf16/2-gpu/"
    elif args.model == "meta-llama/Llama-3.1-70B-Instruct" and args.n_gpus == 4:
        engine_dir = "llama/70B/trt_engines/bf16/4-gpu/"
    elif args.model == "meta-llama/Llama-3.1-70B-Instruct" and args.n_gpus == 8:
        engine_dir = "llama/70B/trt_engines/bf16/8-gpu/"
    else:
        assert False
    
    # Prepare tensorrt-llm runner from engine
    runner_kwargs = {
        'engine_dir': engine_dir,
        'rank': tensorrt_llm.mpi_rank(),
        # Add other parameters as necessary
    }
    runner = runner_cls.from_dir(**runner_kwargs)
    
    # Inference configurations related to inputs
    # batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    batch_sizes = [128]
    input_output_lengths = [(19, 58)]
    # input_output_lengths = [(161, 388), (19, 58), (756, 200)]
    
    # Lists for results
    latency_list = []
    throughput_list = []
    input_length_list = []
    output_length_list = []
    batch_list = [] 
    
    # Derive tje name of model
    splitted_model_name = args.model.split("/")
    if splitted_model_name[-1] == "":
        model_name = splitted_model_name[-2]
    else:
        model_name = splitted_model_name[-1]
        
    
    for input_length, output_length in input_output_lengths:
        # Prepare the input tokens
        tmp_input_tokens = tokenizer.encode(prompt)
        assert len(tmp_input_tokens) >= input_length
        final_prompt = tokenizer.decode(tmp_input_tokens[:input_length], skip_special_tokens=True)
        
        # Update runner's attributes related to input/output tokens
        runner.max_input_len = input_length
        runner.max_seq_len = input_length + output_length
        
        for batch_size in batch_sizes:
            log_path = "./logs/%s_I_%d_O_%d_B_%d_%dgpus.txt" %(model_name, input_length, output_length, batch_size, args.n_gpus)
            
            print("Experiment for %s" %log_path)
            f = open(log_path, 'w')
            
            # Make batched prompt
            batched_prompt = []
            for _ in range(batch_size):
                batched_prompt.append(final_prompt)
                
            iteration_times = []
            try:
                for iter in range(warmup + n_iterations):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    
                    if iter == warmup + n_iterations - 1:
                        torch.cuda.cudart().cudaProfilerStart()
                
                    # Prepare the input tokens for TensorRT
                    input_tokens = tokenizer.batch_encode_plus(batched_prompt, return_tensors="pt", padding=False)
                    assert len(input_tokens["input_ids"]) == batch_size
                    assert len(input_tokens["input_ids"][0]) == input_length
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
                        # Add other parameters for the generation
                    ).cpu()
                    output_sentences = tokenizer.batch_decode(output_tokens.squeeze(), skip_special_tokens=True)
                    
                    if iter == warmup + n_iterations - 1:
                        torch.cuda.cudart().cudaProfilerStop()
                        
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    
                    iteration_times.append(end - start)
                    
            except Exception as error:
                f.write("Error name: %s\n" %type(error).__name__)
                f.write("\n")
                f.write("Error message:\n")
                f.write(str(error))
                f.close()
                latency_list.append(0) # msec
                throughput_list.append(0)
                input_length_list.append(input_length)
                output_length_list.append(output_length)
                batch_list.append(batch_size)
                continue
            
            duration = sum(iteration_times[-n_iterations:])/n_iterations
            total_new_tokens_generated = batch_size * output_length
            throughput = total_new_tokens_generated / duration
            
            f.write("Results:\n")
            f.write(output_sentences[0])
            f.write("\n")
            f.write("Batch size: %d / Input length: %d / output length: %d\n" %(len(batched_prompt), len(input_tokens[0]), len(output_tokens[0])))
            f.write("Iteration itmes: ")
            for i in range(len(iteration_times)):
                f.write("%f, " %iteration_times[i])
            f.write("\n\n")
            f.write("Latency (msec), Throughput (tokens/sec)\n")
            f.write("%f, %f" %(duration * 1000, throughput))
            f.close()
            
            latency_list.append(duration * 1000) # msec
            throughput_list.append(throughput)
            input_length_list.append(input_length)
            output_length_list.append(output_length)
            batch_list.append(batch_size)
            
            for i in output_tokens:
                assert i.shape[1] == output_length + input_length
                
    assert len(latency_list) == len(input_output_lengths) * len(batch_sizes)
    assert len(throughput_list) == len(input_output_lengths) * len(batch_sizes)
    assert len(input_length_list) == len(input_output_lengths) * len(batch_sizes)
    assert len(output_length_list) == len(input_output_lengths) * len(batch_sizes)
    assert len(batch_list) == len(input_output_lengths) * len(batch_sizes)
    
    final_path = "./logs/final_results_%s_%dgpus.txt" %(model_name, args.n_gpus)
    
    f = open(final_path, 'w')
    f.write("Input_len, output_len, batch_size, latency (msec), throughput (token/sec)\n")
    for i in range(len(batch_list)):
        f.write("%d, %d, %d, %f, %f\n" %(input_length_list[i], output_length_list[i], batch_list[i], latency_list[i], throughput_list[i]))
    f.close()
    
    
    
if __name__ == "__main__":
    run()
