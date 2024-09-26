#!/bin/bash

# Input Arguments
type=$1      # linear or power
start=$2     # Starting value
end=$3       # Ending value
stride=$4    # Stride value
input_size=$5
method=$6
bench=$7

# Check if required arguments are provided
if [[ -z "$type" || -z "$start" || -z "$end" || -z "$stride" || -z "$input_size" || -z "$method" || -z "$bench" ]]; then
    echo "Usage: $0 <type: linear|power> <start> <end> <stride> <dtype>"
    exit 1
fi

# Generate values based on type (linear or power)
if [ "$type" == "linear" ]; then
    index_sizes=()
    for ((i=start; i<=end; i+=stride)); do
        index_sizes+=($i)
    done
elif [ "$type" == "power" ]; then
    index_sizes=()
    value=$start
    while [ $value -le $end ]; do
        index_sizes+=($value)
        value=$((value * stride))
    done
else
    echo "Invalid type. Please choose 'linear' or 'power'."
    exit 1
fi

echo "Generated values for index sizes: ${index_sizes[@]}"

# Run the loop for M, K, N values
for index_size in "${index_sizes[@]}"
do
    file_name="${bench}_${method}_input_${input_size}_index_${index_size}.txt"
	output_file="./logs/${file_name}"
    echo $output_file
	if [ -f "$output_file" ]; then
        echo "File $output_file already exists. Skipping..."
        continue
    fi

    echo $input_size, $index_size
    python gather_scatter.py --input-size ${input_size} --output-size ${index_size} --method ${method} --bench ${bench} > ${output_file}
done