#!/bin/bash

# Input Arguments
type=$1      # linear or power
start=$2     # Starting value
end=$3       # Ending value
stride=$4    # Stride value
fixed_size=$5
method=$6
bench=$7

# Check if required arguments are provided
if [[ -z "$type" || -z "$start" || -z "$end" || -z "$stride" || -z "$fixed_size" || -z "$method" || -z "$bench" ]]; then
    echo "Missing shell script arguments"
    exit 1
fi

# Generate values based on type (linear or power)
if [ "$type" == "linear" ]; then
    variables=()
    for ((i=start; i<=end; i+=stride)); do
        variables+=($i)
    done
elif [ "$type" == "power" ]; then
    variables=()
    value=$start
    while [ $value -le $end ]; do
        variables+=($value)
        value=$((value * stride))
    done
else
    echo "Invalid type. Please choose 'linear' or 'power'."
    exit 1
fi

echo "Generated values for index sizes: ${variables[@]}"

if [ "$bench" == "gups_gather" || "$bench" == "gather_s" || "$bench" == "gather_v" ]; then
    echo "Generated values for index sizes: ${variables[@]}"
elif [ "$bench" == "gups_update" || "$bench" == "scatter_s" || "$bench" == "scatter_v" ]; then
    echo "Generated values for index sizes: ${variables[@]}"
fi

# Run the loop for M, K, N values
for variable in "${variables[@]}"
do
    file_name="${bench}_${method}_input_${fixed_size}_index_${variable}.txt"
	output_file="./logs/${file_name}"
    echo $output_file
	if [ -f "$output_file" ]; then
        echo "File $output_file already exists. Skipping..."
        continue
    fi

    echo $fixed_size, $variable

    if [ "$bench" == "gups_gather" || "$bench" == "gather_s" || "$bench" == "gather_v" ]; then
        python gather_scatter.py --input-size ${fixed_size} --output-size ${variable} --method ${method} --bench ${bench} > ${output_file}
    elif [ "$bench" == "gups_update" || "$bench" == "scatter_s" || "$bench" == "scatter_v" ]; then
        python gather_scatter.py --input-size ${variable} --output-size ${fixed_size} --method ${method} --bench ${bench} > ${output_file}
    fi      
done