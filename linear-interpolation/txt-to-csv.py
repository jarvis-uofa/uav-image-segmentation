import csv

# Define the names for your input and output files
input_filename = 'rgb.txt'
output_filename = 'rgb.csv'

# Open the input text file for reading and the output CSV file for writing
with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
    # Initialize the CSV writer
    csv_writer = csv.writer(outfile)

    # Write the header row to the CSV file
    # This gives each column a name for easier processing later
    csv_writer.writerow(['x', 'y', 'width', 'height'])

    # Loop through each line in the input text file
    for line in infile:
        # 1. Strip leading/trailing whitespace from the line
        # 2. Split the line into a list of values based on spaces
        row_values = line.strip().split()
        
        # Ensure the row has the expected number of values (4) before writing
        if len(row_values) == 4:
            # Write the list of values as a new row in the CSV file
            csv_writer.writerow(row_values)

print(f"Successfully converted '{input_filename}' to '{output_filename}'.")

