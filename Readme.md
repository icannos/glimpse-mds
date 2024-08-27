
This is the repositotry of  GLIMPSE: Pragmatically Informative Multi-Document Summarization for Scholarly Reviews
[Paper](https://arxiv.org/abs/2406.07359) | [Code](https://github.com/icannos/glimpse-mds)


### Installation
- We use python 3.10 and CUDA 12.1
- First, create a virtual environment using:
``` bash
conda create -n glimpse python=3.10
```

- Second, install pytorch via the following command:
``` bash
conda install pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
- Finally, all remaining required packages could be installed with the requirements file:

``` bash
pip install -r requirements
```


`rsasumm/` provides a python package with an implementation of RSA incremental decoding and RSA reranking of candidates.
`mds/` provides the experiment scripts and analysis for the MultiDocument Summarization task.


## Citation

If you use this code, please cite the following papers:

```@misc{darrin2024glimpsepragmaticallyinformativemultidocument,
      title={GLIMPSE: Pragmatically Informative Multi-Document Summarization for Scholarly Reviews}, 
      author={Maxime Darrin and Ines Arous and Pablo Piantanida and Jackie CK Cheung},
      year={2024},
      eprint={2406.07359},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.07359}, 
}
```