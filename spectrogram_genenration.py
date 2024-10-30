from utils import *

dataset = 'urbansound'
export = True

if __name__ == "__main__":
    if dataset == 'audio_mnist':
        audio_data = load_data_mnist('data/audioMNIST/data')
        specs = gen_spectgrams_mnist(audio_data, n_fft=256, padding_length=6457) # 6457 is the length of the longest audio file after trimming
        
        if export:
            output_folder = "data/audioMNIST/data_spec"
            os.makedirs(output_folder, exist_ok=True)

            # Iterate through the results and save each spectrogram matrix as a .npy file
            i = 0
            for spec_db, file_name in specs:
                if i % 100 == 0:
                    print(f"Saving file {i}/{len(specs)}")
                output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.npy")
                np.save(output_path, spec_db)
                i += 1
    if dataset == 'urbansound':
        audio_data = load_data_urban('data/audio/')
        specs = gen_spectgrams_urban(audio_data, n_fft=2048) 
        
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