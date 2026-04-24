import os
import numpy as np
import pretty_midi

def process_midi_to_pianoroll(midi_path, sequence_length=256, fs=4):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = midi_data.get_piano_roll(fs=fs).T
        piano_roll[piano_roll > 0] = 1
        
        sequences = []
        for i in range(0, piano_roll.shape[0] - sequence_length, sequence_length):
            seq = piano_roll[i:i + sequence_length]
            if seq.shape[0] == sequence_length:
                sequences.append(seq)
        return sequences
    except Exception as e:
        return []

def build_dataset(raw_data_dir, output_file, max_files=1000):
    all_sequences = []
    processed_count = 0
    
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                file_path = os.path.join(root, file)
                seqs = process_midi_to_pianoroll(file_path)
                
                if seqs:
                    all_sequences.extend(seqs)
                    processed_count += 1
                    
                if processed_count >= max_files:
                    break
        if processed_count >= max_files:
            break

    dataset = np.array(all_sequences)
    print(f"Processed {processed_count} files.")
    print(f"Final dataset shape: {dataset.shape}")
    np.save(output_file, dataset)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    print("Script started!")
    if not os.path.exists('./data/raw_midi/'):
        print("ERROR: Python cannot find the './data/raw_midi/' folder!")
    else:
        files_found = sum([len(files) for r, d, files in os.walk('./data/raw_midi/') if any(f.endswith('.mid') for f in files)])
        print(f"Found {files_found} MIDI files in the folder. Starting processing...")
        
        build_dataset(
            raw_data_dir='./data/raw_midi/', 
            output_file='./data/processed/clean_midi_dataset.npy', 
            max_files=2500 
        )
    print("Script finished completely!")
