# Final Model
## Steps
1. Read data
2. Find correlation
3. Choose an Algorithm; in this case Random Forest
4. Find best training parameters
5. Pass the training parameters to cross validation model

## Running Model
* Trained and Tested on GCP's cloud engine (Special feature for our Project)
* ![alt text](https://i.imgur.com/IFj4xRr.png)
* Run model at: <http://35.195.47.27:5000/tree/AI_CLASS/models#>. (Please Ask me to start the instance before you run)

### Setting up your GCP:
* GCP gives you $300 free credit/year
* Tutorial: <https://cs231n.github.io/gce-tutorial/>

### Alternatively, deploy to now (512Mb, 1vCPU at most) - not 52GB
* Sign up on [ZEIT](https://zeit.co/now)
* Get a token [here](https://zeit.co/account/tokens)
* Click this: [![Deploy to now](https://deploy.now.sh/static/button.svg)](https://deploy.now.sh/?repo=https://github.com/K-2SO-VADER/Heart-Disease-Prediction/tree/jupyter)
* Visit the /_src path given to you to get your jupyter token in the logs
* Use the token to login to jupyter

#### OR Deploy to GCP and ZEIT(now),then perform benchmarks of your own