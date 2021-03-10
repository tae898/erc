Backprop is done to minimize the training cross entropy loss, but the model
with the highest validation f1_weighted is chosen.
Speakers are in caps (e.g. JOEY instead of Joey).
Two utterances are chosen that are back to back (a sequence of two sequences!).
The MELD dataset is noisy. The two utterances that are supposed to be back 
to back aren't always back to back. But nonethelsss I don't change them and 
use what the dataset authors have provided us.

For example:
<s>JOEY: Hi how are you?</s></s>CHANDLER: You wanna go for a drink?</s>

Notice that there are two separation tokens between the two utterances.
Facebook (the original authors of RoBERTa) did this way so I do this way too.

<s> is always prepended to the entire sequence. It works as a BOS and CLS.
