# Understanding Data Leakage in Federated Learning for Acoustic Signals
This is the repository that has been used for this bachelor thesis, all relevant code are provided to reproduce our results.

### Overview of files
* iDLG_adjusted.py is the main algorithm which is a modified version of the one used in its respective paper.
* utils.py are general utility functions such as the FFT and STFT algorithm, padding of signals, and data loader for Audio MNIST and Urbansound 8K
* utils_model.py are functions utilized in the main algorithm including network architecture, normalization, data loader, etc.
* spectrogram_generation is a simple script that generates a spectrogram for the dataset specified.
* griffin_lim.py is the modified fast griffin-lim algorithm found at: https://github.com/rbarghou/pygriffinlim.
* results.py is the reconstruction of waveforms from produced spectrogram matrices.

*Note folder locations etc. should be created, as the creation of correct folders is not included in the code.* 

Retrieved spectrograms from the iDLG algorithm, their respective output files and reconstructed waveforms can be found at: https://www.dropbox.com/scl/fo/v737ivvkr44xq74srqjbr/AO-x6c4Toebl6CFlLo3GH18?rlkey=yu5etu36x8r8pv17rfpsrp2l5&st=fexdo4xb&dl=0 (Available until 21/12)  
