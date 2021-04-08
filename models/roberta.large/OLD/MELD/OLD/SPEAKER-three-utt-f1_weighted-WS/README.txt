Backprop is done to minimize the training cross entropy loss, but the model
with the highest validation f1_weighted is chosen.
Speakers are prepended in caps (e.g. JOEY instead of Joey)
Three utterances are chosen that are in a row.
The MELD dataset is noisy. The utterances that are supposed to be in a row 
aren't always in a row. But nonethelsss I don't change them and 
use what the dataset authors have provided us.

For example:
<s>CHANDLER: Haven't seen you in a while! JOEY: Hi how are you?</s></s>CHANDLER: You wanna go for a drink?</s>

Since RoBERTa expects two segments in one input sequence, I shaped the three
utterances as above. The first two utterances make the first segment and the third
utterance is the second segment. The goal is to predict the emotion of the last 
(third) utterance, which is the current speaker emotion.

If there are no past utterances available (e.g. the current speaker emotion is 
the first utterance), the past utterance(s) are simply a white space (i.e. ' ').

For example, to predict the emotion of the first utterance, it is shaped as:

<s>'  '</s></s>CHANDLER: Haven't seen you in a while!</s>

Notice that the above sequence has two white spaces since I wanted to be consistent
(i.e. the model will see two white spaces as two past utterances.). Honestly,
I don't think this matters so much but let's just go with it.

Also notice that there are two separation tokens (i.e. </s></s>). This is how 
facebook and others have trained their models. In their code you'll see something
like this:

Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`

I think the reason behind this is that if </s> happens once at a time, it only 
happens once at the end of the sequence. And if it happens twice in a row, it 
only happens between the two segments. In this way the neural network learns 
how we shape the input.

<s> is always prepended to the entire sequence. It works as a BOS and CLS.


real training data:
   
  CHANDLER: also I was the point person on my company’s transition from the KL-5 to GR-6 system.
CHANDLER: also I was the point person on my company’s transition from the KL-5 to GR-6 system. THE INTERVIEWER: You must’ve had your hands full.
