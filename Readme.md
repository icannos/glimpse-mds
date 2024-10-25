
This is the repositotry of  GLIMPSE: Pragmatically Informative Multi-Document Summarization for Scholarly Reviews
[Paper](https://arxiv.org/abs/2406.07359) | [Code](https://github.com/icannos/glimpse-mds)


### Installation

- We use python 3.10 and CUDA 12.1
``` bash
module load miniconda/3
module load cuda12
```
- First, create a virtual environment using:
``` bash
conda create -n glimpse python=3.10
```

- Second, install pytorch via the following command:
``` bash
conda install pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
- To activate the environment:
``` bash
conda activate glimpse 
```
- Finally, all remaining required packages could be installed with the requirements file:

``` bash
pip install -r requirements
```
### Data Loading

Step 1: First start by processing the input files from data.

``` bash
python glimpse/data_loading/data_processing.py 
```

Step 2: In this step, we generate candidate summaries.
- for extractive candidates, use the following command:
``` bash
python glimpse/data_loading/generate_extractive_candidates.py 
```
- for abstractive candidates, use the following command:
``` bash
python glimpse/data_loading/generate_abstractive_candidates.py 
```

### RSA Computing
To compute the rsa score for each candidate summary generated in step 2:
``` bash
python python glimpse/src/compute_rsa.py --summaries data/candidates/[Name_Of_Your_File_Step2].csv
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