<div align="center">
  
# AIBugHunter Replication Package
  
</div>

<p align="center">
  </a>
  <h3 align="center"><a href="https://aibughunter.github.io/">AIBugHunter</a></h3>
  <p align="center">
    A Practical Tool for Predicting, Classifying and Repairing Software Vulnerabilities
  </p>
</p>

## Table of contents

<!-- Table of contents -->
<details open="open">
  <summary></summary>
  <ol>
    <li>
      <a href="#how-to-replicate">How to replicate</a>
        <ul>
          <li><a href="#about-the-environment-setup">About the Environment Setup</a></li>
          <li><a href="#about-the-datasets">About the Datasets</a></li>
          <li><a href="#about-the-experiment-replication">About the Experiment Replication</a></li>
        </ul>
    </li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
    </li>
    <li>
      <a href="#license">License</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## How to replicate 

  
### About the Environment Setup
<details open="open">
  <summary></summary>
  
First of all, clone this repository to your local machine and access the main dir via the following command:
```
git clone https://github.com/awsm-research/AIBugHunter.git
cd AIBugHunter
```

Then, install the python dependencies via the following command:
```
...
```

Alternatively, we provide requirements.txt with version of packages specified to ensure the reproducibility,
you may install via the following commands:
```
pip install -r requirements.txt
```

If having an issue with the gdown package, try the following commands:
```
git clone https://github.com/wkentaro/gdown.git
cd gdown
pip install .
cd ..
```

* We highly recommend you check out this <a href="https://pytorch.org/">installation guide</a> for the "torch" library so you can install the appropriate version on your device.
  
* To utilize GPU (optional), you also need to install the CUDA library, you may want to check out this <a href="https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html">installation guide</a>.
  
* <a href="https://www.python.org/downloads/release/python-397/">Python 3.9.7</a> is recommended, which has been fully tested without issues.
 
</details>
  
### About the Datasets
<details open="open">
  <summary></summary>
We use the Big-Vul dataset provided by Fan et al., for more information about the dataset, please refer to [this repository](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset). 
</div>
</details>

### About the Experiment Replication 
<details open="open">
  <summary></summary>

  We recommend to use **GPU with 8 GB up memory** for training since **BERT architecture is very computing intensive**. 
  
  Note. If the specified batch size is not suitable for your device, 
  please modify **--eval_batch_size** and **--train_batch_size** to **fit your GPU memory.**
  
### How to reproduce RQ1 (CWE-ID Classification)
Run the following command to retrain our approach
```
cd rq1_cwe_id_cls/mo_bert
sh train.sh
```

### How to replicate RQ2 (CWE-Type Classification)

    
### How to replicate RQ3 (CVSS Score Regression)
Run the following command to retrain our approach
```
cd rq3_cvss_score_reg/bert
sh train.sh
```
  
</details>


## Acknowledgements
* Special thanks to dataset providers of Big-Vul (<a href="https://dl.acm.org/doi/10.1145/3379597.3387501">Fan et al.</a>)

## License 
<a href="https://github.com/awsm-research/AIBugHunter/blob/main/LICENSE">MIT License</a>

## Citation
```bash
under review
```
