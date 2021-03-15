# Utility functions, I'll split them into different files
# if this one starts becoming too big

from random     import choice
from glob       import glob
from os         import remove, replace
from subprocess import run
from settings   import PATH_RELS, DATASET
from os.path    import isfile

# Counts the number of lines in a file in a quickly manner
def count_file_lines(path):
    with open(path) as f:
        for i, _ in enumerate(f, start=1): pass
    return i


# Generates random negative examples from a list of positive triples
# and the possible targets for each relation
def generate_negatives(triples, targets):
    res = []

    for s, r, t in triples:
        candidates = targets[r]
        if len(candidates) < 2: continue

        rnd_t = choice(candidates)
        while rnd_t == t:
            rnd_t = choice(candidates)

        # We add the label here, and the original positive
        res.append(((s, r, rnd_t, 0), (s, r, t)))

    return res


# Removes the negative examples from a file (the training one). 
# We don't nede them since we generate our own negatives
def remove_negatives_train(file_path):
    keep_feats = []
    with open(file_path, "r") as f:
        for line in f:
            spl = line.strip().split("\t")
            if len(spl) == 3 or spl[3] == "1":
                keep_feats.append(line)
    
    with open(file_path, "w") as f:
        f.writelines(keep_feats)


# Join the csv files that each thread generates
# Additionally, it adds the corresponding header.
def join_files(joint_file, pattern):
    file = open(joint_file, "a")

    with open(PATH_RELS, "r") as f:
        relations = [x.strip().split("\t")[0] for x in f.readlines()]

    from features import get_header

    rels_to_study = None
    rels_study_path = f"datasets/{DATASET}/relations_to_study.txt"
    if isfile(rels_study_path):
        rels_to_study = []
        with open(rels_study_path, "r") as f:
            for line in f:
                if line:
                    rels_to_study.append(line.strip().split("\t")[0])

    header = ";".join(get_header(relations, rels_to_study))
    file.write(header + "\n")

    for part_file in glob(pattern):
        with open(part_file, "r") as f:
            file.write(f.read())
        remove(part_file)

    file.close()


# Get the paths of a certain length between a source and a target entity
# given a list of triples
def get_paths(triples, source, target, length):
    paths = []
    paths_aux = []

    # First iteration
    paths = [[(s, r, t)] for s, r, t in triples if t == target]

    # Subsequent iterations
    for _ in range(1, length):
        for p in paths:
            triples_connect = [(s, r, t) for s, r, t in triples if t == p[0][0]]
            paths_aux = [[x] + p for x in triples_connect]

        paths = paths_aux

    # Return only those whose first entity is the source entity
    paths_filtered = [p for p in paths if p[0][0] == source]
    return paths_filtered

# Filter out the useless features that always have the same values
# Note that the features to be filtered are selected using the training csv,
# but they are deleted from both
def filter_features(train_path, test_path):
    inds_same = None
    first_line = []

    with open(train_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0: continue  # Skip the header
            spl = line.strip().split(";")

            if inds_same is None:
                inds_same = [i for i in range(len(spl))]
                first_line = spl
            else:
                to_del = []
                for i in inds_same:
                    if spl[i] != first_line[i]:
                        to_del.append(i)
                for i in to_del:
                    inds_same.remove(i)

    keep_feats = [i not in inds_same for i in range(len(first_line))]
    _trim_features(train_path, keep_feats)
    _trim_features(test_path, keep_feats)

def filter_contexts(context_size, original_path, new_path):
    not_ctxs = {
        1: ["c3", "c2"],
        2: ["c3"],
        3: [],
    }[context_size]

    with open(original_path, "r") as f:
        for line in f:
            header = line.strip().split(";")
            break
    
    keep_feats = [not any(c in x for c in not_ctxs) for x in header]
    _trim_features(original_path, keep_feats, new_path)

# Aux function to keep only a certain amount of columns
def _trim_features(path, keep, new_file=None):
    aux_path = path + ".aux"
    aux_handle = open(aux_path, "a")

    with open(path, "r") as f:
        for line in f:
            spl = line.strip().split(";")
            filt = [x for i, x in enumerate(spl) if keep[i]]
            aux_handle.write(";".join(filt) + "\n")

    aux_handle.close()
    new_path = new_file if new_file else path
    replace(aux_path, new_path)


def split_file_rels(original_path, handles):
    header = None
    with open(original_path, "r") as f:
        for line in f:
            if not header:
                header = line
                for handle in handles.values():
                    handle.write(header)
            else:
                rel = line.strip().split(";")[0].split(",")[1]
                handles[rel].write(line)

# This one is highly reliant on the environment but it's quick and it works well
def create_compressed_file():
    run(["tar", "cfv", f"{DATASET}.tbz2", "--use-compress-prog=pbzip2", "train/", "test/"],
        cwd=f"output/{DATASET}")