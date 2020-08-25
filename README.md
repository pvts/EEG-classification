# EEG-classification

This code was used for the purposes of a group assignment (max 3 members, with Hamza Khan & Isabella Dintinjana) for the course in "Deep Learning" of the programme MSc Data Science at the University of Tilburg. The results of our predictions, and those of the other groups, were uploaded on Codalab. This assignment was our first attempt at ML and DL and it was submitted on **23/03/2020**, therefore it is reflective of our skills and knowledge at the time of submission.

# Dataset Description 

The data-set consists of EEG sequences of 3000-time steps each and coming from two electrode locations on the head (Fpz-Cz and Pz-Oz) sampled at 100 Hz. That means that each sample contains two signals of 3000 samples and that those samples correspond to 30 seconds of recording. The labels that come along with the data specify six stages labelling them with corresponding numbers.

Each sequence in the dataset contains only one stage, which is speciﬁed by the corresponding label. The evaluation of the final model was based on an unlabeled test set that was provided to us. 

The data set is presented in two different formats, Raw signals and Spectrograms.
1. The ﬁle "Data_Raw_signals.pkl" contains the sequences and the corresponding labels as two array [sequences, labels].
2. The ﬁle "Data_Spectrograms.pkl" contains the spectrograms of the sequences and the corresponding labels as two array [spectrograms, labels].

In the ﬁle "Data_Spectrograms.pkl" the spectrograms have a size 100 by 30 for each signal, and they represent the same 3000-time steps EEG sequences as in the raw data. That is, for each sequence in the raw data ﬁle, there is a corresponding spectrogram. The spectrograms in "Data_Spectrograms.pkl" represent the frequencies of the signals in steps of 0.5Hz between 0.5 and 50 Hz (Hence 100). Such frequencies correspond to the spectral information in time windows of size 100-time steps each; thus, for a sequence of 3000-time steps, there are 30 windows (hence spectrogram size: 100 by 30, frequencies by the number of windows).

Note - Due to the university's policy and regulations, the dataset cannot be made publicly available.

# Evaluation Metric
Classification Accuracy - My team achieved an accuracy of 67.45% on the test set.
