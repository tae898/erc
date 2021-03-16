Backprop is done to minimize the training cross entropy loss, but the model
with the highest validation f1_weighted is chosen.
Speakers are in title (e.g. Joey instead of JOEY).
Two utterances are chosen that are back to back (a sequence of two sequences!).
The MELD dataset is noisy. The two utterances that are supposed to be back 
to back aren't always back to back. But nonethelsss I don't change them and 
use what the dataset authors have provided us.

For example:
<s>Joey: Hi how are you?</s></s>Chandler: You wanna go for a drink?</s>

Notice that there are two separation tokens between the two utterances.
Facebook (the original authors of RoBERTa) did this way so I do this way too.

<s> is always prepended to the entire sequence. It works as a BOS and CLS.


```python
for _ in range(history):
    utterances.insert(0, '')

for _ in range(history):
    utterances.pop()
```

If there is no past utterance available (i.e. the first utterance in a dialogue),
the past utterance is just ''. It's nothing.

```python
tokens = roberta.encode(*['Rachel: Hi'])
tokens.tolist()
[0, 41415, 35, 12289, 2]

tokens = roberta.encode(*['', 'Rachel: Hi'])
tokens.tolist()
[0, 2, 2, 41415, 35, 12289, 2]
```