#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <file_to_transfer> <nodes> <ppn>"
    exit 1
fi

# Assign the input filename, nodes, and ppn
filename="$1"
nodes="$2"
ppn="$3"
filename_without_extension="${filename%.c}"  # Remove the .c extension

# Destination directory on the server
destination_directory="parallel-CNN-MPI"

# SFTP transfer to remote server
sftp_command="put $filename $destination_directory"
password="972420380129"
sshpass -p "$password" sftp e5332-04@192.168.99.137 <<<"lcd $(dirname "$filename")"$'\n'"$sftp_command"$'\n'"bye"


# SSH to remote server, compile, and execute
# compile_command="/opt/mpich2/gnu/bin/mpicc -I/home/e5332-04/ $filename -o $filename_without_extension -lm"
# execute_command="qsub -l nodes=$nodes:ppn=$ppn ./qsub_script"
# sshpass -p "$password" ssh e5332-04@192.168.99.137 "$compile_command && $execute_command" 2>&1 | tee result_output.txt

# # Echo the result on the local terminal
# echo "Execution on remote server completed."

