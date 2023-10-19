import csv
import time
import os

# Initialize a buffer to store the results
buffer = []

# Set the maximum buffer size
max_buffer_size = 1000

# Set the interval at which to flush the buffer to the file (in seconds)
flush_interval = 10


# Start a timer to keep track of the last flush
last_flush_time = time.time()

def add_result_to_buffer(result):
    # Add the result to the buffer
    buffer.append(result)
    
    # Check if the buffer size exceeds the maximum
    if len(buffer) >= max_buffer_size:
        # Flush the buffer to the file
        flush_buffer()

    # Check if the flush interval has passed
    current_time = time.time()
    if current_time - last_flush_time >= flush_interval:
        # Flush the buffer to the file
        flush_buffer()
        last_flush_time = current_time

def flush_buffer():
    # Open the file in append mode
    os.makedirs('./results', exist_ok=True)
    with open(f'./results/result_{time.strftime('%Y-%m-%d__%H_%M_%S')}.csv', 'a', newline='') as f:
        # Create a CSV writer
        writer = csv.DictWriter(f,fieldnames=)
        
        # Write each result in the buffer to the file
        for result in buffer:
            writer.writerow(result)
        
        # Clear the buffer
        buffer.clear()