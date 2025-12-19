import pyttsx3

engine = pyttsx3.init()
input_sentence = input("Enter the sentence: ")
engine.say(input_sentence)
engine.runAndWait()
