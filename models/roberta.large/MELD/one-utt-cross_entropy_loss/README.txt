Optimized to minimize the validation loss.
No Speakers added to the utterances.
This is exactly what the COSMIC authors did. The cosmic authors claimed that
the test score was 62.02 but I got 64.396, although I did exactly what they did.
Maybe they didn't tune the hyper parameters correctly.

Only one speaker utterance used.

For example:
<s>Hi how are you?</s>

<s> is always prepended to the entire sequence. It works as a BOS and CLS.
