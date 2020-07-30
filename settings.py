from multiprocessing import cpu_count
import sys

N_THREADS = 8  # Maybe set it to cpu_count()
DIRECTIONAL_GRAPH = True

try:
    DATASET = sys.argv[1]
except IndexError:
    print("Please provide a dataset as a command line argument")
    sys.exit()

MAX_CONTEXT_SIZE = 2 if "NELL" in DATASET else 3

PATH_TRAIN = f"datasets/{DATASET}/train.txt"
PATH_TEST = f"datasets/{DATASET}/test.txt"
PATH_RELS = f"datasets/{DATASET}/relations.txt"
