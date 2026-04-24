import numpy as np
import pretty_midi

def pianoroll_to_midi(piano_roll, output_file_path, fs=4):
    """
    Convert a piano roll (numpy array) back to a MIDI file.
    piano_roll shape: (T, 128) where T is the sequence length.
    Values should be 1.0 (note ON) or 0.0 (note OFF).
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    
    # Iterate over the 128 possible notes
    for note_num in range(128):
        # find where the note is pressed
        is_playing = piano_roll[:, note_num] >= 0.5
        
        # find the start and end indices of continuous note presses
        changes = np.diff(is_playing.astype(int))
        starts = np.where(changes == 1)[0] + 1
        if is_playing[0]:
            starts = np.insert(starts, 0, 0)
            
        ends = np.where(changes == -1)[0] + 1
        if is_playing[-1]:
            ends = np.append(ends, len(is_playing))
            
        # Create Note objects for each continuous press
        for start, end in zip(starts, ends):
            start_time = float(start) / fs
            end_time = float(end) / fs
            note = pretty_midi.Note(velocity=100, pitch=note_num, start=start_time, end=end_time)
            instrument.notes.append(note)
            
    midi.instruments.append(instrument)
    midi.write(output_file_path)

if __name__ == "__main__":
    pass
