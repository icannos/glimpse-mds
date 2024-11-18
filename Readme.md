
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
- Second, activate the environment and install pytorch:
``` bash
conda activate glimpse 
conda install pytorch==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

- Finally, all remaining required packages could be installed with the requirements file:

``` bash
pip install -r requirements
```
### Data Loading

Step 1: Start by processing the input files from data.

``` bash
python glimpse/data_loading/data_processing.py 
```

### Generating Summaries and Computing RSA Scores
Step 2: Now, we generate candidate summaries and compute RSA scores for each candidate
- for extractive candidates, use the following command:
``` bash
sbatch scripts/extractive.sh Path_of_Your_Processed_Dataset_Step1.csv
```
- for abstractive candidates, use either of the following commands:
  - In case the last batch is incomplete, you can add padding using `--add-padding` argument to complete it:
  ``` bash
  sbatch scripts/abstractive.sh Path_of_Your_Processed_Dataset_Step1.csv --add-padding
  ```
  - If you want to remove the last incomplete batch, you can run the script without the argument:
  ``` bash
  sbatch scripts/abstractive.sh Path_of_Your_Processed_Dataset_Step1.csv
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