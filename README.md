# Reloop-RP-8000-RASPI-BPM
Send BPM from audio in from the Reloop RP8000 mk1 or mk2 to the display using a raspberry pi with a audio soundcard connected

# Installation

Clone repo and cd into it 

activate your virtualenv (optional)

```
pip3 install -r requirements.txt
```

# Usage
```
python3 bpm_detection.py --filename /tmp/test.wav --window 3
```

Special thanks to 
https://github.com/ChasonDeshotel/Reloop-RP-8000-MIDI (for reverse engineering the RP8000 midi)
https://github.com/scaperot/the-BPM-detector-python (for the bpm detector)