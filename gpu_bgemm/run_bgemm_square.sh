#!/bin/bash

# Input Arguments
type=$1      # linear or power
start=$2     # Starting value
end=$3       # Ending value
stride=$4    # Stride value
fixed_size=$5
method=$6
dtype=$7

# Check if required arguments are provided
if [[ -z "$type" || -z "$start" || -z "$end" || -z "$stride" || -z "$method" || -z "$dtype" || -z "$fixed_size" ]]; then
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
    file_name="bgemm_${method}_b_${variable}_m_${fixed_size}_k_${fixed_size}_n_${fixed_size}_dtype_${dtype}.txt"

	output_file="./logs/${file_name}"
    echo $output_file
	if [ -f "$output_file" ]; then
        echo "File $output_file already exists. Skipping..."
        continue
    fi

    echo "M, K, N: $fixed_size, B: $variable, dtype: $dtype"
    cmd="python bgemm.py --B ${variable} --M ${fixed_size} --K ${fixed_size} --N ${fixed_size} --dtype ${dtype} --method vtrain"
    echo $cmd
    ${cmd} > ${output_file}
done