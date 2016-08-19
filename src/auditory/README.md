## Emotive Speech codebase documentation
## Introduction
The process of transforming a normal sounding speech to a speech that sounds happy, sad or afraid is described in this paper which can be found at this link [1], involves at its core frequency shifting of the speech signal. For example, to transform a neutral speech to a happy sounding speech one has to first shift the frequency of each sound chunk by a certain amount-a process known as pitch shifting and again shift the frequency of the first 0.5 sec. of an utterance by a certain amount- a process known as inflection.
This emotive speech codebase, uses an algorithm known as TD-PSOLA to implement frequency shifting of a sound signal according to these papers and a book[2][3][4]. TD-PSOLA contains three major steps. After analyzing the pitch of each chunk of sound, we implement the first step of TD-PSOLA called pitch-marking, which involves two phases [3]. After pitch-marking, we implement the second step of TD-PSOLA, which is called the synthesis phase [2][4].

#### Below, I list and describe the python modules that implement TD-PSOLA and the transformation of neutral speech to an emotive one.
* ## voiced_unvoiced.py
    This module is used to analyze an audio signal and its various features – finds fundamental frequencies of chunks, classifies voiced and unvoiced regions, creates a dictionary of voiced and unvoiced regions starting points, creates an array of lists of starting and ending points of voiced chunks amongst other functions.
* ## Inflection.py
    This module is used to implement inflection as described in this paper [1]. There are two functions that implement inflection to create two emotions- fear and happiness. They are inflection fear and inflection_happy_newest_two respectively.
* ## pitch_mark_first_step.py
    As described in this book [2] the first stage of frequency shifting using TD-PSOLA algorithm is pitch-marking. According to this paper [3] pitch-marking involves two phases – the first phase is implemented by this module.
* ## pitch_mark_second_stage.py
    According to this paper [3] pitch-marking involves two phases – the second phase is implemented by this module. The method optimal_accumulated_log_probability returns an object that contains all the best pitch marks within a given sound array.

* ## td_psola.py
    As described in this book [2] the second stage of frequency shifting using TD-PSOLA algorithm is the synthesis phase. This module specifically freq_shift_using_td_psola_newest method synthesizes the sound array to a new sound array with the given frequency shift. 
* ## freq_shift_array.py
    This module contains functions that create arrays which specify the amount by which a given chunk of sound’s fundamental frequency should be shifted - the names given are good indictors of the kinds of arrays they create. 
* ## ems.py
    This is a dictionary of words and the emotions they correspond to. 
* ## EmotiveSpeech.py
    This is a module that contains a method with the same name, which accepts the name of a certain emotion and an audio  file name– converting that audio file to an audio that evokes that specified emotion. 
* ## convertSpeech.py
    This is a module that contains a method with the same name, which accepts a text filename and an audio filename  –converting the audio generated by a text to speech converter that reads that text, to a speech that evokes the emotion that an analyzer deems to fit the text and saving the speech in the audio filename specified.



## Usage
As an end user, one only needs to focus on two modules within this codebase. These are EmotiveSpeech.py and convertSpeech.py. If the user wants to convert a
pre-recorded speech audio file to a speech file with a desired emotion attached to it, she/he has to use the method found in EmotiveSpeech.py. If the user wants to convert a text to speech with a corresponding emotion attached to it, she/he has to use the method found in convertSpeech.py.
## Comments and recommendation for future developers
* The file naming system used in this library conforms to a windows os.
* The processing speed of the codebase is relatively slow mainly because it is written in Python. It would be better for this feature to be written in C/C++
* For higher frequency shifts there is some noticeable jitter.

## References
* [1] http://biorxiv.org/content/early/2016/02/01/038133
* [2] DAFX - Digital Audio Effects, Udo Zolzer, pages 222-226 
* [3]https://www.researchgate.net/publication/221488978_A_two_phase_pitch_marking_method_for_TD-PSOLA_synthesis
* [4] cslab1.bc.edu/~csacademics/pdf/13kim.pdf