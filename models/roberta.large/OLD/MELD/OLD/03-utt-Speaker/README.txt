Finally figured out how things work. From here on, I'll clean every utterance
before shaping them into a sequence. I don't know how much dirty utterances
affect the performance (probably small), but my OCD didn't allow them. 

The model with the lowest cross entropy loss on the validation split was chosen,
but of course backprop is done from the training split. Since every dataset and 
split has different data distributions, it's difficult to choose a metric for 
the validation split. That's why I chose the most generic one, cross entropy loss.

This method uses three utterances in one sequence input. The first two are the
past two utterances, which makes the first segment, and the third is the current
speaker utterance, which makes the second segment. The two segments are separated
by the separation tokens. The goal is to predict the correct emotion of the second
segment, which is the current speaker utterance. If there are no past utterances
available, then the first segment is just an empty segment.

Speakers are prepended in title (e.g. Joey instead of JOEY). 

--------------------------------------------------------------------------------

Here are some examples:

1. Past two utterances are available:

<s>Joey: Hey Chandler, how are you? Chandler: I'm doing good!</s></s>Joey: That's nice to hear!</s>

2. One past utterance is available:

<s>Chandler: I'm doing good!</s></s>Joey: That's nice to hear!</s>

3. No past utterances are available:

<s></s></s>Joey: That's nice to hear!</s>

--------------------------------------------------------------------------------

Notice that every sequence starts with <s>, which is a BOS token. This only
happens once in every sequence. Between the two segments, there are two </s>
tokens. If this happens twice in a row, it means that they separate the two
segments. If this only happens once, not two in a row, it only happens at the end
of the sequence, which works as an EOS token. The pretrained RoBERTa models were
pretrained this way, so it's important to follow the guidelines in order to fully
take advantage of transfer learning.
