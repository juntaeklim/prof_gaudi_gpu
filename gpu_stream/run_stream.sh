#!/bin/bash

# Input Arguments
type=$1      # linear or power
start=$2     # Starting value
end=$3       # Ending value
stride=$4    # Stride value
method=$5
bench=$6

# Check if required arguments are provided
if [[ -z "$type" || -z "$start" || -z "$end" || -z "$stride" || -z "$method" || -z "$bench" ]]; then
    echo "Usage: $0 <type: linear|power> <start> <end> <stride> <dtype>"
    exit 1
fi

# Generate values based on type (linear or power)
if [ "$type" == "linear" ]; then
    input_sizes=()
    for ((i=start; i<=end; i+=stride)); do
        input_sizes+=($i)
    done
elif [ "$type" == "power" ]; then
    input_sizes=()
    value=$start
    while [ $value -le $end ]; do
        input_sizes+=($value)
        value=$((value * stride))
    done
else
    echo "Invalid type. Please choose 'linear' or 'power'."
    exit 1
fi

echo "Generated values for input sizes: ${input_sizes[@]}"

# Run the loop for M, K, N values
for input_size in "${input_sizes[@]}"
do
    file_name="${bench}_${method}_input_${input_size}.txt"
	output_file="./logs/${file_name}"
    echo $output_file
	if [ -f "$output_file" ]; then
        echo "File $output_file already exists. Skipping..."
        continue
    fi

    echo $input_size
    python stream.py --input-size ${input_size} --method ${method} --bench ${bench} > ${output_file}
done