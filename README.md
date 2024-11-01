## Learning to construct sequential events

Code for the preprint 'Consolidation of sequential experience into a deep generative network explains human memory, prediction and planning'.

#### Installation

To install and run the code:

```bash
git clone https://github.com/ellie-as/sequence-memory.git
cd sequence-memory
pip install -r requirements.txt
```

Then open the relevant Jupyter notebook (see below).

The code was tested with Python 3.8.10, with trainng run on a single A100. To train models on MacOS, add --use_mps_device whenever the run_clm_*.py script is run.

#### Subfolders

* **narratives**: code for showing how the model neocortex learns the gist of specific events in Bartlett (1932) and Raykov et al. (2023), reconstructing them with gist-based distortions.
* **statistical learning**: code for Durrant et al. (2011) simulation.
* **inference**: code for exploring relational inference in a spatial and family tree task based on Whittington et al. (2020), and for showing how RAG supports inference from recent memory.
* **planning**: code for simulating the development of model-based planning abilities in Vikbladh et al. (2024) through consolidation.
* **misc**: Code for representing memories as gists and details, and for sequence compression results in the SI.
* **hopfield**: Code for simulations of asymmetric Hopfield networks in SI.
* **scripts**: Helper scripts and utils, including the model training scripts (which are adapted from the HuggingFace Trasformers Python library training script examples).

