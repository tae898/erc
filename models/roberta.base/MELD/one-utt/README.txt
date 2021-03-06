Optimized to minimize the validation loss.
No Speakers added to the utterances.
This is exactly what the COSMIC authors did, except that their model is
three times larger than mine (They used RoBERTa-large).
Only one speaker utterance used.

For example:
<s>Hi how are you?</s>

<s> is always prepended to the entire sequence. It works as a BOS and CLS.
