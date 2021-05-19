# CAFE: Knowledge Graph Completion using Neighborhood-Aware Features [![DOI:10.1016/j.engappai.2021.104302](https://zenodo.org/badge/DOI/10.1016/j.engappai.2021.104302.svg)](https://doi.org/10.1016/j.engappai.2021.104302)

Source code for the CAFE tool, which evaluates triples for KG Completion. 

This repository contains the necessary code to generate feature vectors using our proposed context-aware features, as well as the datasets that we used in our paper. To run CAFE, install the dependencies listed in `requirements.txt` and run `python main.py <dataset> <max-ctx>`, where `<dataset>` is the name of the dataset to use (which should be in the `datasets/` folder) and `<max-ctx>` is the maximum path length used to generate neighborhood subgraphs.

If you find CAFE useful, please consider citing it as:

```bibtex
@article{borrego2021CAFE,
    author = {Borrego, Agust{\'i}n and Ayala, Daniel and Hern{\'a}ndez, Inma and Rivero, Carlos R. and Ruiz, David},
    title = {{CAFE}: Knowledge graph completion using neighborhood-aware features},
    journal = {Engineering Applications of Artificial Intelligence},
    volume = {103},
    pages = {104302},
    year = {2021},
    issn = {0952-1976},
    doi = {10.1016/j.engappai.2021.104302},
    url = {https://www.sciencedirect.com/science/article/pii/S0952197621001500}
}
```