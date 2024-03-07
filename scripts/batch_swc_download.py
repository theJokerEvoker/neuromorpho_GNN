import os
import requests
import argparse

# List for holding links
file_links = []

# Failed files for some reason
failed_files = []

# Define dictionary to map parameters to file names
file_mapping = {
    "human": "filter_human.txt",
    "droso": "filter_droso.txt",
    "rat": "filter_rat.txt",
    "mouse": "filter_mouse.txt"
}

# Create argument parser for file
parser = argparse.ArgumentParser(description="Parse HTML and obtain direct links to .swc files")

# Add required parameter argument
parser.add_argument("--dataset", type=str, choices=file_mapping.keys(), required=True, help="Choose a dataset: human, droso, rat, mouse")

# Parse command-line arguments
args = parser.parse_args()

# Use provided parameter to determine the file name
file_name = file_mapping[args.dataset]

# Parse HTML and format to obtain direct links to .swc files
with open(file_name, 'r') as file:
    for line in file:
        if line.startswith('<img alt='):
            split1 = line.split('value=\"')
            lab_name = split1[1].split('\"', 1)[0].lower()
            neuron_name = split1[2].split('\"', 1)[0]

            if "," in lab_name:
                lab_name = lab_name.split(',')[0]
            
            link_name = 'https://neuromorpho.org/dableFiles/' + lab_name + '/CNG%20version/' + neuron_name + '.CNG.swc'
            file_links.append(link_name)

for link in file_links:
    r = requests.get(link, stream=True)

    if r.ok:
        file_name = link.split('version/', 1)[1]

        with open(file_name, 'wb') as swc:
            for chunk in r.iter_content(chunk_size=1024*8):
                if chunk:
                    swc.write(chunk)
                    swc.flush()
                    os.fsync(swc.fileno())
    else:
        failed_files.append(link)
        continue

with open("failed_files.txt", "a") as file:
    for link in failed_files:
        file.write(str(link) + '\n')