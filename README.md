# Vocal Style Transfer
A project for transferring music vocal style from one song to another.

## What is the goal?
This project aims to transfer the singing style from one song to another. For example, imagine listening to a song with Elton John's voice with the same music, but with a different tone of voice!
Theoretically, there is no limit to the destination sound! You can set your voice to any song you like with the original singer's tone.
This project has only implemented the section related to style transfer. Before providing the main singer's voice and the target voice, the original singer's voice track must be separated from the music. Other models such as [Deezer Spleeter](https://github.com/deezer/spleeter) can be used for this purpose.

## About the codes
   1. The idea has been inspired by various articles, some of which are mentioned in the references section. However, the existing code was not copied from any of these sources and was developed from scratch.
   2. The link related to the relevant data is placed in the codebase.

## Training notes
For model training, it is highly recommended to use V100 and P100 GPU series or train the model with TPU. Otherwise, the training time will be very long.

## References
- [Voice Conversion Using Speech-to-Speech Neuro-Style Transfer](https://ebadawy.github.io/post/speech_style_transfer/)
- [Symbolic Music Genre Transfer with CycleGAN](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer)
- [Groove2Groove: One-shot style transfer for music accompaniments](https://groove2groove.telecom-paris.fr/demo.html)
- [Audio texture synthesis and style transfer](https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/)
- [Global Prosody Style Transfer Without Text Transcriptions](https://auspicious3000.github.io/AutoPST-Demo/)
- [Voice style transfer with random CNN](https://github.com/mazzzystar/randomCNN-voice-transfer)
