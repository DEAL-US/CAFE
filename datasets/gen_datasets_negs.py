from os import listdir
from os.path import isdir
from random import choice

n_negs = 10

for dset in listdir("."):
    if not isdir(dset): continue
    print(dset)

    print("Reading train")
    cands_ls = {}
    train_triples = {}
    with open(f"{dset}/train.txt", "r") as f:
        for line in f:
            s, p, o, label = line.strip().split("\t")
            if label != "1": continue
            cands_ls[p] = cands_ls.get(p, []) + [o]
            train_triples[(s,p,o)] = 1

    cands = {k: list(set(v)) for k, v in cands_ls.items()}

    print("Reading test")
    tpls = {}
    with open(f"{dset}/test.txt", "r") as f:
        for line in f:
            s, p, o, label = line.strip().split("\t")
            if label == "1":
                tpls[(s,p,o)] = 1

    print("Writing test")
    handle = open(f"{dset}/test.txt", "w")
    for s, p, o in tpls:
        handle.write(f"{s}\t{p}\t{o}\t1\n")
        os = [o_cand for o_cand in cands[p]
         if (s, p, o_cand) not in train_triples and (s, p, o_cand) not in tpls]

        for _ in range(n_negs):
            if not os: break
            o_cand = choice(os)
            os.remove(o_cand)
            handle.write(f"{s}\t{p}\t{o_cand}\t-1\n")
    handle.close()