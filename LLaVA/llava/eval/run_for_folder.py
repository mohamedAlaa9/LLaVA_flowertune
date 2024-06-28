import os
import subprocess
import argparse
import csv

def main(folder, query, model_path, temperature, output_csv):
    # List all files in the folder
    image_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    # Open CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File Path", "Result"])

        # Iterate over each image file and run the original script
        for image_file in image_files:
            image_path = os.path.join(folder, image_file)
            command = [
                "python", "LLaVA_flowertune/LLaVA/llava/eval/run_llava.py",
                "--image-file", image_path,
                "--query", query,
                "--model-path", model_path,
                "--temperature", str(temperature)
            ]

            # Run the command and capture the output
            result = subprocess.run(command, capture_output=True, text=True)

            # Write the file path and result to the CSV file
            writer.writerow([image_path, result.stdout.strip()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Folder containing the images")
    parser.add_argument("--query", type=str, required=True, help="Query to pass to the script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for the model")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    main(args.folder, args.query, args.model_path, args.temperature, args.output_csv)
