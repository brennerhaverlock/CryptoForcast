# CryptoForcast

![alt text](https://raw.githubusercontent.com/officialbrenner/CryptoForcast/master/BitcoinTestPredictions.png)


This is designed to forcast cryptocurrency prices using neural network of type LSTM(RNN network) with tensorflow. Currently I'd only "trust" the original Cryptoforcast.py as it's for Bitcoin and it's the best looking prediction at the moment. 



## Usage

```$ git clone https://github.com/officialbrenner/CryptoForcast```

### 1. Install required packages
I use Anaconda to handle my packages
```
$ conda install -c conda-forge tensorflow 
$ conda install -c anaconda numpy 
$ conda install -c conda-forge matplotlib 
$ conda install -c conda-forge/label/broken matplotlib 
$ conda install -c conda-forge/label/testing matplotlib 
$ conda install -c conda-forge/label/rc matplotlib
$ conda install -c anaconda scikit-learn 
```
### 2. Get API key and create config file

https://www.alphavantage.co/support/#api-key

Get your API key and create a new file called config and create a variable named api_key 

$ api_key = 'YOUR_API_KEY'

### 3. Run. 
Run program using your IDE(I recommend Spyder). Check out this post(https://medium.com/@brenner.haverlock/variables-that-matter-for-rnn-lstm-networks-329d422c58a9) for help. 
