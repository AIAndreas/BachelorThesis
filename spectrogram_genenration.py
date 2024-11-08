from utils import *

dataset = 'urbansound'
export = True

if __name__ == "__main__":
    if dataset == 'audio_mnist':
        audio_data = load_data('data/mnist')
        specs = gen_spectgrams(audio_data, max_signal_length=47998, n_fft=512) # 6457 is the length of the longest audio file after trimming
        
        if export:
            output_folder = "data/mnist/data_spec"
            os.makedirs(output_folder, exist_ok=True)

            # Iterate through the results and save each spectrogram matrix as a .npy file
            i = 0
            for spec_db, _, file_name in specs:
                if i % 100 == 0:
                    print(f"Saving file {i}/{len(specs)}")
                output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.npy")
                np.save(output_path, spec_db)
                i += 1
    if dataset == 'urbansound':
        audio_data = load_data('data/audio/')
        specs = gen_spectgrams(audio_data, max_signal_length=192000, n_fft=1024) 
        
        if export:
            output_folder = "data/audio/data_spec"
            os.makedirs(output_folder, exist_ok=True)

            # Iterate through the results and save each spectrogram matrix as a .npy file
            i = 0
            for spec_db, _ , file_name in specs:
                if i % 100 == 0:
                    print(f"Saving file {i}/{len(specs)}")
                output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.npy")
                np.save(output_path, spec_db)
                i += 1