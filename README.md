
# Correlation Analysis in Multi-Modal Datasets
A python implementation of the Techniques presented in '[1]' for model selection and correlation structure estimation in multiple datasets.
Given a multi-modal dataset, this technique estimates the following:
* The number of correlated components across the datasets.
* The structure of the correlated components

The technique solves the complete model selection problem shown above by employing bootstrap based hypothesis testing.

Cite the work as follows:

```
@article{hasija2019determining,
  title={Determining the dimension and structure of the subspace correlated across multiple data sets},
  author={Hasija, Tanuj and Lameiro, Christian and Marrinan, Timothy and Schreier, Peter J},
  journal={arXiv preprint arXiv:1901.11366},
  year={2019}
}

```

## Installation

To install the toolbox and the required packages, (it is recommended to create a virtual environment) simply run:
```
git clone https://github.com/praneeth-b/Correlation-Analysis-in-High-Dimensional-Data.git

cd Correlation-Analysis-in-High-Dimensional-Data/

python setup.py install

```

## Repository Structure

* `multipleDatasets.algo` provides the implentation of the algorithms presented in [1].
* `multipleDatasets.simulateData` provides objects to create a correlation structure and create synthetic data for the prescribed correlation structure
* `multipleDatasets.visualization` provides objects to visualize the correlation structures as graphs.
* `multipleDatasets.examples` provides a tutorial style jupyter notebook which demonstrates the usage of this toolbox.


## References
[1] T. Hasija, C. Lameiro, T. Marrinan,  and P. J. Schreier,"Determining the Dimension and Structure of the Subspace Correlated Across Multiple Data Sets,".

[2] T. Hasija, Y. Song, P. J. Schreier and D. Ramirez, "Bootstrap-based Detection of the Number of Signals Correlated across Multiple Data Sets," Proc. Asilomar Conf. Signals Syst. Computers, Pacific Grove, CA, USA, November 2016.


