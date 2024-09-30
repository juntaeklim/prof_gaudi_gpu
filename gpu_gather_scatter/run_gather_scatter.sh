#!/bin/bash

# Input Arguments
type=$1      # linear or power
start=$2     # Starting value
end=$3       # Ending value
stride=$4    # Stride value
fixed_size=$5
method=$6
bench=$7
dim_size=$8

# Check if required arguments are provided
if [[ -z "$type" || -z "$start" || -z "$end" || -z "$stride" || -z "$fixed_size" || -z "$method" || -z "$bench" || -z "$dim_size" ]]; then
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

echo "Fixed size: $fixed_size"
echo "Variables ${variables[@]}"

# Run the loop for M, K, N values
for variable in "${variables[@]}"
do
    if [[ "$bench" == "gather_v" || "$bench" == "gather_s" || "$bench" == "gups_gather" || "$bench" == "gups_update" ]]; then
        file_name="${bench}_${method}_input_${fixed_size}_output_${variable}_dim_${dim_size}.txt"
    elif [[ "$bench" == "scatter_v" || "$bench" == "scatter_s" ]]; then
        file_name="${bench}_${method}_input_${variable}_output_${fixed_size}_dim_${dim_size}.txt"
    fi

	output_file="./logs/${file_name}"
    echo $output_file
	if [ -f "$output_file" ]; then
        echo "File $output_file already exists. Skipping..."
        continue
    fi

    if [[ "$bench" == "gups_update" || "$bench" == "gups_gather" || "$bench" == "gather_s" || "$bench" == "gather_v" ]]; then
        echo "input_size: $fixed_size, output_size: $variable"
        python gather_scatter.py --input-size ${fixed_size} --output-size ${variable} --dim-size ${dim_size} --method ${method} --bench ${bench} --using-prepared-indices --algo randint > ${output_file}
    elif [[ "$bench" == "scatter_s" || "$bench" == "scatter_v" ]]; then
        echo "input_size: $variable, output_size: $fixed_size"
        python gather_scatter.py --input-size ${variable} --output-size ${fixed_size} --dim-size ${dim_size} --method ${method} --bench ${bench} --using-prepared-indices --algo randint > ${output_file}
    fi
done