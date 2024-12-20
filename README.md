# Dual-path-RNN, Conv-TasNet

Before run train and inference, setup vertiual env.
```
git clone https://github.com/SBerik/DualPathRNN.git
cd ./DualPathRNN
python3 -m venv .venv 
```
For Linux (Ubuntu):
```
source .venv/bin/activate
```
For Window: 
```
.\.venv\Scripts\activate
```
Review requirements.txt and then install it:
```
pip install -r requirements.txt
```
To run `train.py`. Before running: review the config.yml files
```
python train.py -p ./configs/NAME_OF_YOUR_CONFIG_FILE.yml
```
Warining: Be careful where you run the script

To run `inference.py`
```
python inference.py PATH_TO_SAMPLE -plot -s -t THRESHOLD -c checkpoints/NAME_OF_YOUR_CONFIG_FILE -w weights/WEIGHT_FILE_NAME
```
Warning: run the file `inference.py` with the trained model weights and the config file. 

## Data
The dataset used for this project is the **Kazakh Speech Corpus**, which is available at [https://issai.nu.edu.kz/kz-speech-corpus/](https://issai.nu.edu.kz/kz-speech-corpus/). This dataset consists of audio recordings in Kazakh language. The KS2 contains around 1.2k hours of high-quality transcribed data comprising over 600k utterances.

Scheme of data folder:
```
data/
    annotation/ 
        Csv/
            Dev/
            Test/
            Train/
        Json/
            Dev/
            Test/
            Train/
    KSC2/
        Dev/
        Test/
        Train/
    samples/
        5ed8a1c0f3ea2.flac
        5ed8a1c0f3ea2.csv
```

## Hardware 
Used GPU: `NVIDIA GeForce RTX 4090`  
Number of GPUs: `1`  
CUDA version: `cu118`