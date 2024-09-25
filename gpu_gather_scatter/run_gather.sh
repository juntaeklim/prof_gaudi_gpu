#!/bin/bash

# Input Arguments
type=$1      # linear or power
start=$2     # Starting value
end=$3       # Ending value
stride=$4    # Stride value
input_size=$5
method=$6
v_or_s=$7

# Check if required arguments are provided
if [[ -z "$type" || -z "$start" || -z "$end" || -z "$stride" || -z "$input_size" || -z "$method" || -z "$v_or_s" ]]; then
    echo "Usage: $0 <type: linear|power> <start> <end> <stride> <dtype>"
    exit 1
fi

# Generate values based on type (linear or power)
if [ "$type" == "linear" ]; then
    output_sizes=()
    for ((i=start; i<=end; i+=stride)); do
        output_sizes+=($i)
    done
elif [ "$type" == "power" ]; then
    output_sizes=()
    value=$start
    while [ $value -le $end ]; do
        output_sizes+=($value)
        value=$((value * stride))
    done
else
    echo "Invalid type. Please choose 'linear' or 'power'."
    exit 1
fi

echo "Generated values for output sizes: ${output_sizes[@]}"

# Run the loop for M, K, N values
for output_size in "${output_sizes[@]}"
do
    file_name="gather_${v_or_s}_${method}_input_${input_size}_output_${output_size}.txt"
	output_file="./logs/${file_name}"
    echo $output_file
	if [ -f "$output_file" ]; then
        echo "File $output_file already exists. Skipping..."
        continue
    fi

    echo $input_size, $output_size
    python gather_scatter.py --input-size ${input_size} --output-size ${output_size} --method ${method} --bench gather_${v_or_s} > ${output_file}
done