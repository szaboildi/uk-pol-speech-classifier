
This repository is for training classifier for UK parliamentary speeches, which guesses what party's member gave the given speech in the UK House of Commons. You can find an API based on this repo [here](https://svm6-pvutvs4yla-ew.a.run.app) and a frontend [here](https://pol-speech-classifier.streamlit.app/) based on this [Github repo](https://github.com/szaboildi/uk-pol-speech-classifier-frontend).

# Install
The codebase for the training and preprocessing can be found in the `polclassifier/` folder along with some registry functions (for saving and loading models). Currently only SVM and KNN models are supported by this package, as these were found to be the best performing ones. After cloning the repo, it can be installed locally with ```pip install -e .``` The repo also contains scripts for building and deploying a Docker image.

# The Classifier
## Preprocessing and Data
The model is trained on data from the [ParlSpeech V2 corpus](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN), which contains a total of 500,000 speeches from the UK House of Commons given between 1988 to 2019. This dataset was limited to speeches with at least 400 words in them, and limited to the 7 parties that had at least 1000 such speeches in the corpus (list in the Task section). To balance the dataset, each class was downsampled to 1000 observations, leading to a training data set of 7000 speeches. Longer speeches were cut down to 600 words, which were extracted from the middle of the speech

## The Task
The model is a 7-way classifier between the following parties: Conservative Party, Labour Party, Liberal Democrats, SNP, DUP, UUP and Plaid Cymru. It also returns a confidence or prediction probability along with the predicted class.

## Best-performing Model
The repository is set up to train our best-performing model by default using sklearn. The model is an SVM with a linear kernel, C of 1.32 and gamma set to `"scale"`, using a TFIDF vectorized input using `gensim`'s `"glove-wiki-gigaword-100"` with a `min_df` of 5, a `max_df` of 85% and 10000 features. It reaches 60.79% accuracy across the 7 classes but some classes are predicted better than others. Regional parties (SNP, DUP, UUP and Plaid Cymru) are predicted with over 70% accuracy (even 78% on UUP), whereas nation-wide parties (Conservative, Labour and LibDem) are often mixed up with one-another and the accuracy hovers around 36-47%.

## Performance comparisons
The model performs better than the other options that have been tried, these can be found in the `notebooks/` folder. Explored models include a KNN (43.78% accuracy), a vanilla Logistic Regression (59.95% accuracy), an LSTM (49.07% accuracy), a GRU (51.71% accuracy) and a transformer (with [BERT-small](https://huggingface.co/prajjwal1/bert-small) used for the encoder, 46.40% accuracy). Currently only the SVM and KNN are packaged along with some additional registry functions for keras-based models.
