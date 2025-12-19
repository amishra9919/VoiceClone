from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from SV.encoder import inference as encoder
from TTS.synthesizer.inference import Synthesizer
from TTS.vocoder import inference as vocoder
from SV.encoder.audio import trim_long_silences
import noisereduce as nr
from tkinter import *
from tkinter import filedialog
from translate import Translator
import sounddevice as sd
from tkinter import messagebox
import threading
import queue

window = Tk()

###Variables
InputLanguageChoice = StringVar()
TranslateLanguageChoice = StringVar()

LanguageChoices = {"English","Afrikaans", "Hmong", "Indonesian","Japanese","Somali"}
InputLanguageChoice.set('English')
TranslateLanguageChoice.set('English')

recording = False
file_exists = False    
q = queue.Queue()

def CloneMYui():
    if not browsefile_bool:
        messagebox.showwarning("Warning", "Please browse and select an audio file first")
        return
    if not OutputVar.get():
        messagebox.showwarning("Warning", "Please translate text first")
        return
    
    enc_model_fpath = Path("SV/encoder/new_saved_models/encoder.pt")
    syn_model_fpath =Path("TTS/synthesizer/saved_models/pretrained/pretrained.pt")
    voc_model_fpath = Path("TTS/vocoder/saved_models/pretrained/pretrained.pt")
    encoder.load_model(enc_model_fpath)
    vocoder.load_model(voc_model_fpath)
    loop_condition = True
    loop_number = 0
    max = 0
    while loop_condition :
            if loop_number == 0:
                if browsefile_bool :
                    in_fpath = filename 
                    #audio_file : converts stereo to mono (normalized to a range between -1 and 1, amplitude of the sound wave at each sample)
                    #sampling_r is samples per second (Librosa resamples to 22050 Hz by default unless specified)
                    audio_file, sampling_r = librosa.load(str(in_fpath)) 
                    #preprocess_wav returns with the resampling and normalize the audio
                    preprocessed_wav = encoder.preprocess_wav(audio_file, sampling_r) 
                else:
                    audio_file = filename_browseFiles
                    preprocessed_wav = encoder.preprocess_wav(audio_file, sr) 
       ############################################################################################################
            loop_number += 1
            if loop_number == 9:
                break
        ########################################## Embeddings #####################################################
            #Hours of speech of audio is convert into embedding .. using this embeddings can generate desired audio
            #it finds features like pitchpatterns, vocal tract shape, speaking style, timbre and resonance 
            #usually it is 256 or 512 dimensions
            embed = encoder.embed_utterance(preprocessed_wav)
            global embeds
            embeds = [embed]
            text = Translation 
        ############################################# TTS ##########################################################
            global synthesizer
            synthesizer = Synthesizer(syn_model_fpath)
            texts = [text]
            specs = synthesizer.synthesize_spectrograms(texts, embeds) #convert to mel spectogram from text and audio
            global generated_wav
            generated_wav = vocoder.infer_waveform(specs)
            generated_wav = encoder.preprocess_wav(generated_wav)
        ############################################################################################################
            output_audio_duration = librosa.get_duration(y=generated_wav, sr=synthesizer.sample_rate)
            reduced_noise = nr.reduce_noise(y=generated_wav, sr=synthesizer.sample_rate)
            trim = trim_long_silences(reduced_noise)
            trimmed_audio_duration = librosa.get_duration(y=trim, sr=synthesizer.sample_rate)
            threshold_to_break = 0.9
            ratio = trimmed_audio_duration / output_audio_duration
            if max < ratio:
                max = ratio
                global max_wav
                max_wav = generated_wav

            print("ratio",ratio)
            if ratio >= threshold_to_break :
                break
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    num_generated = 0
    output_filename = "cloned_output%02d.wav" % num_generated
    output_path = output_dir / output_filename
    sf.write(str(output_path), generated_wav.astype(np.float32), synthesizer.sample_rate)
    saved.set(str(output_path))
    messagebox.showinfo("Success", f"Voice cloning completed! Audio saved as {output_path}")



def browseFiles():
    global browsefile_bool
    global filename
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select an Audio File",
                                          filetypes=(("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                                                     ("All files", "*.*")))
    if filename:
        browsefile_bool = True
        audio_status.set(f"✓ {Path(filename).name}")
    else:
        browsefile_bool = False
        audio_status.set("No file selected")

def Translate():
    if not TextVar.get():
        messagebox.showwarning("Warning", "Please enter text to translate")
        return
    translator = Translator(from_lang= InputLanguageChoice.get(),to_lang=TranslateLanguageChoice.get())
    global Translation
    Translation = translator.translate(TextVar.get())
    OutputVar.set(Translation)

def threading_rec(x):
    if x == 1:
        
        t1=threading.Thread(target= record_audio)
        t1.start()
    elif x == 2:
        
        global recording
        recording = False

def callback(indata, frames, time, status):
    q.put(indata.copy())

def record_audio():
    
    global sample_rate
    sample_rate = 16000  
    global recording 
    
    recording= True   
    global file_exists 
    
    messagebox.showinfo(message="Recording Audio. Speak into the mic")
    try:
        with sf.SoundFile("trial.wav", mode='w', samplerate=44100,channels=2) as file:
                with sd.InputStream(samplerate=44100, channels=2, callback=callback):
                    while recording == True:
                        file_exists =True
                        file.write(q.get())
    except Exception as e:
        messagebox.showerror("Error", f"Recording failed: {str(e)}")
        recording = False
        return

    if recording == False:
        global audio_file, filename_browseFiles, browsefile_bool, sr
        browsefile_bool = False
        audio_file, sr = librosa.load(str("trial.wav"))
        audio_file = nr.reduce_noise(y = audio_file, sr = sr)
        filename_browseFiles = nr.reduce_noise(y = audio_file, sr = sr)
       



window.title("VOICE CLONE MAKER")
window.geometry("600x500")
window.minsize(600,500)
window.maxsize(600,500)
window.config(bg='light yellow')

# Step 1: Browse Audio
Label(window, text="Step 1: Select Audio File", font=('Cambria',11,'bold'), bg='light yellow').place(x=30, y=20)
Button(window, text='Browse Audio File', bg='white', font=('Cambria',10,'bold'), borderwidth=2, relief='solid', command=browseFiles, width=15).place(x=30, y=50)
audio_status = StringVar()
audio_status.set("No file selected")
Label(window, textvariable=audio_status, font=('Cambria',9), bg='light yellow', fg='blue').place(x=180, y=53)

# Step 2: Input Text
Label(window, text="Step 2: Enter Text", font=('Cambria',11,'bold'), bg='light yellow').place(x=30, y=100)
Label(window, text="Input Language:", font=('Cambria',9), bg='light yellow').place(x=30, y=130)
InputLanguageChoiceMenu = OptionMenu(window, InputLanguageChoice, *LanguageChoices)
InputLanguageChoiceMenu.config(width=10)
InputLanguageChoiceMenu.place(x=140, y=125)
Label(window, text="Enter Text:", font=('Cambria',9), bg='light yellow').place(x=30, y=165)
TextVar = StringVar()
Entry(window, textvariable=TextVar, width=50, font=('Cambria',9)).place(x=140, y=165)

# Step 3: Translate
Label(window, text="Step 3: Translate Text", font=('Cambria',11,'bold'), bg='light yellow').place(x=30, y=210)
Label(window, text="Target Language:", font=('Cambria',9), bg='light yellow').place(x=30, y=240)
NewLanguageChoiceMenu = OptionMenu(window, TranslateLanguageChoice, *LanguageChoices)
NewLanguageChoiceMenu.config(width=10)
NewLanguageChoiceMenu.place(x=140, y=235)
Button(window, text='Translate', bg='white', font=('Cambria',10,'bold'), borderwidth=2, relief='solid', command=Translate, width=12).place(x=270, y=237)
Label(window, text="Translated Text:", font=('Cambria',9), bg='light yellow').place(x=30, y=275)
OutputVar = StringVar()
Entry(window, textvariable=OutputVar, width=50, font=('Cambria',9), state='readonly').place(x=140, y=275)

# Step 4: Clone Voice
Label(window, text="Step 4: Clone Voice", font=('Cambria',11,'bold'), bg='light yellow').place(x=30, y=320)
Button(window, text='Clone Voice', bg='lightgreen', font=('Cambria',11,'bold'), borderwidth=2, relief='solid', command=CloneMYui, width=15).place(x=30, y=350)

# Output
Label(window, text="Output:", font=('Cambria',11,'bold'), bg='light yellow').place(x=30, y=400)
Label(window, text="Saved as:", font=('Cambria',9), bg='light yellow').place(x=30, y=430)
saved = StringVar()
saved.set("Not generated yet")
Label(window, textvariable=saved, font=('Cambria',9), bg='light yellow', fg='green').place(x=100, y=430)

# Optional: Record Audio (bottom right)
Label(window, text="Or Record Audio:", font=('Cambria',9,'bold'), bg='light yellow').place(x=400, y=20)
Button(window, text="Start Recording", command=lambda m=1:threading_rec(m), bg='lightblue', font=('Cambria',9), width=13).place(x=400, y=50)
Button(window, text="Stop Recording", command=lambda m=2:threading_rec(m), bg='lightcoral', font=('Cambria',9), width=13).place(x=400, y=85)

window.mainloop()