# Conv-TasNet, Dual-path-RNN, Sepformer

Before run train and inference, setup vertiual env.
```
$ git clone https://github.com/SBerik/SeparationForKazakhSpeech.git
$ cd ./SeparationForKazakhSpeech
$ python3 -m venv .venv 
```
For Linux (Ubuntu):
```
$ source .venv/bin/activate
```
For Window: 
```
$ .\.venv\Scripts\activate
```
Review requirements.txt and then install it:
```
$ pip install -r requirements.txt
```
To run `train.py`. Before running: review the config.yml files
```
$ python3 train.py -p ./configs/NAME_OF_YOUR_CONFIG_FILE.yml
```
Warining: Be careful where you run the script

To run `inference.py`
```
$ python3 inference.py -m PATH_TO_MIXED_SAMPLE.flac -c configs/NAME_OF_YOUR_CONFIG_FILE -w weights/WEIGHT_FILE_NAME -s PATH_TO_SAVE_AUDIOS
```
Warning: run the file `inference.py` with the trained model weights and the config file. 

## Data
The dataset used for this project is the **Kazakh Speech Corpus**, which is available at [https://issai.nu.edu.kz/kz-speech-corpus/](https://issai.nu.edu.kz/kz-speech-corpus/). This dataset consists of audio recordings in Kazakh language. The KS2 contains around 1.2k hours of high-quality transcribed data comprising over 600k utterances.

## Hardware 
Used GPU: `NVIDIA GeForce RTX 4090`  
Number of GPUs: `1`  
CUDA version: `cu118`