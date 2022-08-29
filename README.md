{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "7baa3d21",
   "metadata": {},
   "source": [
    "# Speakers Comparison\n",
    "\n",
    "On input speeches are provided and we need to define whether they belong to the same person.\n",
    "To answer this question, we train the model that converts each speech into a feature vector. By measuring the cosine similarity between the vectors corresponding to the different speeches, we make a prediction whether the speaker is the same or not."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "56e71cd5",
   "metadata": {},
   "source": [
    "The training is performed on the dataset of clean speeches from [OpenLSR](https://www.openslr.org/resources/12/train-clean-100.tar.gz) as follows\n",
    "1. For each speech, an audio embedding is obtained with the use of [OpenL3](https://openl3.readthedocs.io/en/latest/)\n",
    "2. The classification problem is solved with the embeddings from OpenL3 playing the role of the input layer.\n",
    "3. The layer before the last one is used to construct the feature vectors of speeches and evaluate equal error rate (EER) with the metrics defined by the cosine similarity\n",
    "4. The trained model and the threshold value corresponding to the EER is further used to compare speakers, which were never heard before"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2410e8b7",
   "metadata": {},
   "source": [
    "## Details on the model construction\n",
    "\n",
    "OpenL3 can provide embeddings of the length 512 or 6144 for sounds corresponding to the environment or music. Here we utilze the default values, namely, the content type is set to \"music\" and the embedding size is 6144.\n",
    "\n",
    "To solve the classification problem the fully-connected neural network with one hidden layer of 512 neurons is used.\n",
    "\n",
    "First, one needs to decide how many speeches will be used for each speaker in course of training. For this purpose we have calculated EER for 1,3, and 10 speeches per speaker on a small dataset containing only 3 different persons. The results for 3 and 10 speeches were found to be close to each other, therefore, in what follows, only 3 speeches per speaker are used for model training.\n",
    "\n",
    "The final classification model is trained on 10 speakers. This model without the last layer is saved to further discriminate the speakers.\n",
    "\n",
    "The EER is calculated on the test set which consists of 10% of the total number of speeches in the dataset. For the trained model the EER is 3%\n",
    "\n",
    "The code for this section with comments is presented in [Create_the_model.ipynb](Create_the_model.ipynb)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f79e8a0b",
   "metadata": {},
   "source": [
    "## Details on the speakers comparison part\n",
    "\n",
    "To test the approach, speeches that were never heard by the model were used. Speakers both known and unknown for the model are used.\n",
    "\n",
    "Each speech is divided into slices of 1 second by the OpenL3. To compare two speeches one can use following different approaches:\n",
    "1. Calculate cosine similarity averaged over all possible slices-slices connections with slices belonging to different speeches.\n",
    "2. Average over slices the feature vectors obtained from the model, and calculate cosine similarity between them.\n",
    "3. Average over slices the feature vectors obtained from OpenL3, pass them to the model, and calculate cosine similarity\n",
    "\n",
    "Here we tried the first two options and found that the second variant shows the best performance.\n",
    "\n",
    "Launches showed no mistakes in speakers comparison, indicating that the test set was not extensive enough.\n",
    "\n",
    "The code for this section with comments is presented in [Compare_speakers.ipynb](Compare_speakers.ipynb)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7378b21b",
   "metadata": {},
   "source": [
    "## Possible improvements\n",
    "\n",
    "Speeches presented in [OpenLSR](https://www.openslr.org/resources/12/train-clean-100.tar.gz) are the segments of the audiobooks. The silence can be found in the speeches of each speaker. This silence may make different speakers closer to each other. To avoid the problems which can arise from this fact one can drop these slices.\n",
    "\n",
    "It is also interesting to investigate the performance of the embeddings of length 512, since less disk space is needed to store these files.\n",
    "\n",
    "More tests should be performed on a larger number of speakers. \n",
    "\n",
    "It is interesting to investigate the performance of the model on the noisy data sets from, e.g., phone calls."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84af8379",
   "metadata": {},
   "source": [
    "# Usefull References\n",
    "\n",
    "[1] J.H.L. Hansen and T. Hasan,\n",
    "''Speaker recognition by machines and humans: A tutorial review'' \n",
    "[IEEE Signal processing magazine 32, 74 (2015)](https://www.semanticscholar.org/paper/Speaker-Recognition-by-Machines-and-Humans%3A-A-Hansen-Hasan/c7d244dde874f82e5982e27391251fa66d41de8f#paper-header).\n",
    "\n",
    "[2] D. Sztaho, G. Szaszak, and A. Beke,\n",
    "''Deep learning methods in speaker recognition: a review'',\n",
    "[arXiv:1911.06615](https://arxiv.org/abs/1911.06615).\n",
    "\n",
    "[3] Z. Bai and X.-L. Zhang,\n",
    "''Speaker Recognition Based on Deep Learning: An Overview'',\n",
    "[arXiv:2012.00931](https://arxiv.org/abs/2012.00931)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bb886fbd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
