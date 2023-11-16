# This is the data and code repository for manuscripts submitted to EMSE journal.

## Datasets
We first give the datasets under the datasets folder shown above, these datasets were processed well by previous researchers.

## Code Pre-trained Models
The code pre-trained models used in this paper are all come from HuggingFace, we give the links to download them:

CodeBERT-base: https://huggingface.co/microsoft/codebert-base

CodeT5-base: https://huggingface.co/Salesforce/codet5-base

PLBART-csnet: https://huggingface.co/uclanlp/plbart-csnet

CodeGPT: https://huggingface.co/microsoft/CodeGPT-small-py-adaptedGPT2

CodeGen-base: https://huggingface.co/Salesforce/codegen-350M-mono

We do not give the models shown above since it is too large and easy to obtain.


## Different Embedding Ways
The embedding method can be found in detail in the paper, and this step can be easily implemented. 

We provide code blocks to obtain embeddings by special tokens and average-pooling all codes tokens. 

These code chunks can be put into the pipeline where the pre-trained model is applied to downstream tasks (e.g., the pipeline provided by CodeXGLUE) to obtain code embeddings, provided that the model parameters are first frozen.







#Finally 
We first give links to the datasets used in this paper which are collected from the open-source repository.
Datasets Devign and BigCloneBench for code vulnerability detection and code clone detection
Devign: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection
and the raw dataset comes from the paper Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks. 
Datasets CWE119 and CWE399: https://github.com/CGCL-codes/VulDeePecker
and these two datasets come from the paper Vuldeepecker: A deep learning-based system for vulnerability detection.
BigCloneBench: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench
and the raw dataset comes from the paper Detecting Code Clones with Graph Neural Network and Flow-Augmented Abstract Syntax Tree.
Datasets openstack, qt, platform, gerrit, and go for just-in-time defect prediction: https://github.com/ZZR0/ISSTA21-JIT-DP
and the raw dataset comes from the paper Deep just-in-time defect prediction: how far are we?

