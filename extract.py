def extract_min_value(file_path, start_line, end_line, keyword="Validation set: Average Domain loss"):
    min_value = float('inf')
    min_line_number = -1
    count = 0
    
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            #print(keyword in line)
            if start_line <= line_number <= end_line and keyword in line:
                # Extract the number after the keyword
                try:
                    value = float(line.split()[-1].strip())
                    if value < min_value:
                        min_value = value
                        min_line_number = line_number
                except ValueError:
                    print("Value error")
                    count += 1
                    # Skip lines where the number extraction fails
                    continue
    return min_line_number, min_value

# Example usage:
file_path = "/home/karan/project/FedSamp/nohup_dec_23_1.out"
start_line = 62700  # Replace with your starting line
end_line = 67700    # Replace with your ending line

line_number, lowest_value = extract_min_value(file_path, start_line, end_line)
if line_number != -1:
    print(f"Lowest value: {lowest_value} found at line {line_number}")
else:
    print("No values found in the specified line range.")
