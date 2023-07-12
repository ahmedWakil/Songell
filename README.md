# Songell - A Fantasy RPG name generator
This is a fantasy name generator that uses a character level language model to generate names. The model simply takes
in one character and tries to predict the next character. To generate names, initially, the model is given a start of stream
character. Then we keep sampling the next character, with the current prediction being the input of the prediction, in a loop 
until the model predicts an end of stream character. This ends the loop and we combine the predicted characters in the
sequence to make a name.

\<sos\> H a l e n a \<eos\>

# Some technical details
A character level language model predicts characters of a word as a sequence. In this section I will try to briefly breakdown
some of the technical details of how all of this works.

## Data preparation
All of the names are stored in plain text files, where the name of text file is the category, and the lines in the text file
are the names associated with that category. The first job is to read in these files and create a dictionary where the keys are
the category names and the values are a list of strings that represent the names. During this process the algorithm also keeps 
track of all of the unique characters across all of the files, and also the category names in a list. This will be important 
for representing words as vectors. When the process is finished we will have created these three structures:
1. {"Elf": ["$Solas&", "$Solosolous&", . . .]}
2. ["'", "A", "B", . . .]
3. ["Aasimar", "Elf", . . .]

Note: here I am using \<sos\> = $, and \<eos\> = &, these are fed into the model as any other character. The choice is arbitrary
as long as these do not appear in the actual names.

### One-hot encoding
One-hot encoding is a very common way to represent a set or characters, words or some other higher level construct, as a 
vector. Suppose you have N unique characters, $`C = \{c_1, c_2, . . . , c_N\}`$ and a vector $`\hat{v} = (v_1, v_2, . . . , v_N)`$,
of size $`(1, N)`$. The vector representation of the character $`c_q\in C`$ is where $`v_q = 1`$ and all other elements of 
$`\hat{v}`$ are 0. For example suppose our character set is all of the lowercase alphabets in the english language, 
$`C = \{a,b,...,z\}`$. A one hot vector for the character 'a' is $`\hat{v} = (1, 0,..., 0)`$ of size $`(1,26)`$, because there 
are 26 lowercase letters in the English language, and 'a' appears in the first position. This type of encoding is also used
to represent the categories as vectors.

## Model design
The model takes in the category one-hot vector and character one-hot vector and concatenates them. which is then passed through
a linear layer. This is a design choice you could also pass the combined vector directly into the LSTM layer but to me this seemed
to make the most sense. This was mainly done to try and capture the relationship between the category and the character. The next
layer is the LSTM layer which is responsible for most of the heavy lifting. I will not get into to much detail but the main
advantage of an LSTM is how the three input's interact. You have your hidden state which allows information from the last prediction
to be taken into consideration, but now you also have the cell state. This enables certain information to persist over time in the 
sequence generation. You can learn more in depth about LSTM's in 
<a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">this</a> article.

The next three layer's job us to take the outputs of the LSTM and make predictions. First it is passed through another linear 
layer so that the output matches the size of the input tensor. This is then passed into a dropout layer, which randomly zeros out 
some outputs with some probability p. According to <a href="https://arxiv.org/abs/1207.0580">this</a> paper, "When a large 
feedforward neural network is trained on a small training set, it typically performs poorly on held-out test data.". We are using 
$`p = 0.1`$ here to try and introduce a bit of fuzz to the data to prevent this overfitting. The final step is to pass this 
through a softmax layer. I am using the inbuilt PyTorch LogSoftMax function, which will give us a probability distribution over 
all of our characters. 

<img width="83%" src="https://github.com/ahmedWakil/Songell/blob/main/public/inferencing-model/Model-Architecture.png">

## Training
Training is not that different from training other neural networks, the only real difference is that the thing that the model is
trying to predict is the next character given the current one. For a name such as "Halena", if we were to construct the 
(input, prediction) tuples at each time step we would have, ("H", "a"), ("a", "l"), ("l", "e"), ("e", "n"), ("n", "a"). To do 
this for any random (Category, name) pair, we have to construct input and target vectors for the name.

Suppose you have a name "$`c_1c_2...c_l`$" of length $`l`$, this includes the \<sos\> and \<eos\> characters. Then the input 
characters are $`c_1c_2...c_{l-1}`$, and the target characters are $`c_2c_3...c_l`$. For example for the name "Halena" we would
have:

$ H a l e n a <br>
H a l e n a &

as the input and target respectively. In pytorch you are able to add up all of the losses up from these prediction for a single
name and call backpropagate afterwards, and let PyTorch handle the gradient calculation and parameter optimization.

## Model sampling
Sampling a language model is not always trivial, in fact, if you were to always take the highest probable character you would
end up with very little variety of names. Even though the use of likelihood as training objective leads to high quality models 
for a broad range of language understanding tasks, using likelihood as a sampling objective leads to text that is bland and 
strangely repetitive, this is called text generation. How you choose to sample new sequences has a profound effect on the variety
and quality. You can read more about this in <a href="https://arxiv.org/abs/1904.09751">this</a> paper.

Currently I do not always take the highest likelihood character, instead I randomly choose from the top three highest instead.
Just this change alone had a profound effect on the names being created.

## Future improvements
### Better sampling methods
This is an active area of research for me as I am trying to understand more robust methods like top k, top p and beam search in an effort
to improve the quality and variety of names being created.
### Improve training
I have also started thinking about better ways to improve the training objective to better suit the generation task. One could be
training a discriminator that penalizes the model for generating names that are too unhuman like, or being too bland.

## Libraries used
### Python
1. PyTorch: 2.0.1+cpu
2. matplotlib: 3.7.1
### JavaScript
1. react-spring/web: 9.7.2
2. onnxruntime-web: 1.15.1
3. react: 18.2.0