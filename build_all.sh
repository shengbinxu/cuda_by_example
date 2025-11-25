#!/bin/bash
mkdir -p bin

# Find all .cu files
find chapter* -name "*.cu" | while read source_file; do
    dir=$(dirname "$source_file")
    filename=$(basename "$source_file")
    parent_dir=$(basename "$dir") 
    
    # Determine output name
    if [ "$filename" == "kernel.cu" ]; then
        output_name="$parent_dir"
    else
        output_name="${filename%.*}"
    fi
    
    echo "Compiling $source_file to bin/$output_name..."
    
    # Compile
    nvcc -w -I./common "$source_file" -o "bin/$output_name" -lglut -lGL -lGLU -lpthread
    
    if [ $? -ne 0 ]; then
        echo "Failed to compile $source_file"
    fi
done

echo "Build complete."


