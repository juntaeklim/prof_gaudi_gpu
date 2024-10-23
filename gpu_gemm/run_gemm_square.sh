#!/bin/bash

# Input Arguments
type=$1      # linear or power
start=$2     # Starting value
end=$3       # Ending value
stride=$4    # Stride value
method=$5
dtype=$6

# Check if required arguments are provided
if [[ -z "$type" || -z "$start" || -z "$end" || -z "$stride" || -z "$method" || -z "$dtype" ]]; then
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
    file_name="gemm_${method}_m_${variable}_k_${variable}_n_${variable}_dtype_${dtype}.txt"

	output_file="./logs/${file_name}"
    echo $output_file
	# if [ -f "$output_file" ]; then
    #     echo "File $output_file already exists. Skipping..."
    #     continue
    # fi

    echo "M, K, N: $variable, dtype: $dtype"
    cmd="python gemm.py --M ${variable} --K ${variable} --N ${variable} --dtype ${dtype} --method vtrain"
    echo $cmd
    ${cmd} > ${output_file}
done