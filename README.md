# Songell - A Fantasy RPG name generator

This is a fantasy name generator that uses a character level language model to generate names. The model simply takes
in one character and tries to predict the next character. To generate names, initially, the model is given a start of stream
character. Then we keep sampling the next character, with the current prediction being the input of the prediction, in a loop 
until the model predicts an end of stream character. This ends the loop and we combine the predected characters in the
sequence to make a name.

\<sos\> H a l e n a \<eos\>

## Some technical details

<img width="80%" src="https://github.com/ahmedWakil/Songell/blob/main/public/inferencing-model/Model-Architecture.png">


## Libraries used
### Python
1. PyTorch: 2.0.1+cpu
2. matplotlib: 3.7.1
### JavaScript
1. react-spring/web: 9.7.2
2. onnxruntime-web: 1.15.1
3. react: 18.2.0