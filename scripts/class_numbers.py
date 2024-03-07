import os
import requests
import argparse

# Define a dictionary to map parameters to directory paths
directory_mapping = {
    "human": './data/NeuroMorpho/human',
    "droso": './data/NeuroMorpho/drosophila',
    "rat": './data/NeuroMorpho/rat',
    "mouse": './data/NeuroMorpho/mouse'
}

# Create an argument parser
parser = argparse.ArgumentParser(description="Count class numbers for dataset")

# Add a required --parameter argument
parser.add_argument("--dataset", choices=directory_mapping.keys(), required=True, help="Choose a dataset: human, droso, rat, mouse")

# Parse the command-line arguments
args = parser.parse_args()

# Use the provided parameter to determine the directory
directory = directory_mapping[args.dataset]

# # Counting primary brain regions
# brain_region_list = []
# with os.scandir(directory) as data_folder:
#     for file in data_folder:
#         if file.name.endswith('.swc') and file.is_file():
#             response = requests.get('https://neuromorpho.org/api/neuron/name/' + file.name.split('.swc', 1)[0])

#             if not response.ok:
#                 print('Response not ok: ' + str(file.name))

#             # Parsing requested metadata text to find primary brain region
#             try:
#                 split1 = response.text.split("\"brain_region\":", 1)[1]
#                 split1 = split1.split(",\"", 1)[0]
#                 split1 = split1.strip("\"[]")
#             except:
#                 print('Split not ok: ' + str(file.name))
#                 continue

#             # Add parsed brain region to list
#             brain_region_list.append(split1)

# from collections import defaultdict

# region_count = defaultdict(lambda: 0)

# for region in brain_region_list:
#     region_count[region] += 1

# for key, value in region_count.items():
#     print(f"{key}: {value}")


# Counting primary cell types
cell_type_list = []
possible_classes = ['Glia', 'interneuron', 'long-range non-principal GABAergic', 'principal cell', 'sensory', 'null']
with os.scandir(directory) as data_folder:
    for file in data_folder:
        if file.name.endswith('.swc') and file.is_file():
            response = requests.get('https://neuromorpho.org/api/neuron/name/' + file.name.split('.swc', 1)[0])

            if not response.ok:
                print('Response not ok: ' + str(file.name))

            # Parsing requested metadata text to find primary cell type
            try:
                split1 = response.text.split("\"cell_type\":", 1)[1]
                split1 = split1.split("],\"", 1)[0]
                split1 = split1.split(",")
                split1 = [x.strip('\"[]') for x in split1]

                if split1[0] in possible_classes:
                    cell_class = split1[0]
                elif split1[-1] in possible_classes:
                    cell_class = split1[-1]
                else:
                    for i in split1:
                        if i in possible_classes:
                            cell_class = i
                            break
                    else:
                        print('Class not found: ' + str(file.name))
                        continue

            except:
                print('Split not ok: ' + str(file.name))
                continue

            # Add parsed cell type to list
            cell_type_list.append(cell_class)

from collections import defaultdict

type_count = defaultdict(lambda: 0)

for region in cell_type_list:
    type_count[region] += 1

for key, value in type_count.items():
    print(f"{key}: {value}")
