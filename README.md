# KOMODIS dataset
This is the repository for the paper: "*A Corpus of Controlled Opinionated and Knowledgeable Movie Discussions for Training Neural Conversation Models*". 
The paper can be found here: http://arxiv.org/abs/2003.13342. 

We introduce an augmented dialogue dataset (**K**nowledgable and **O**pinionated **MO**vie **DIS**cussions) that is crowd-sourced and collected with Amazon Mechanical Turk. Each dialogue is based on two feature structures (one for each crowd-worker) about the same movie:

<center><img src="https://fsmt.blob.core.windows.net/komodis/profiles.PNG" width="60%"></center>



### Dialogue examples
For detailed information please check the paper. Below are two dialogue examples.
![example_dialogues](https://user-images.githubusercontent.com/61545955/109512002-f09cd500-7aa3-11eb-8766-3da6e6320485.png)

### Data
We provide the full postprocessed dialogue dataset in *data/dataset.json*.

For explanations on how to read and use the structured data, please check *data/example.json* (will be updloaded soon!).


### Model
We provide a baseline script to train a GPT-2 model with our dataset in PyTorch in *model/*.

To train a model, you have to run the *train.py* script:
````
python train.py --dataset komodis
````
More information regarding additional arguments can be found in the script. Please download the pretrained 
GPT-2 weights from https://github.com/huggingface/transformers and store them in *data/pretrained_models/gpt2/* and
*data/pretrained_weights/tokenizers*. 
