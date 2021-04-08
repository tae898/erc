Alright this didn't work.
So what I wanted to do is

1. Get the pretrained RoBERTa model.
2. Pretrain again on the MELD training dataset using the masked language modeling
3. Finetune on the MELD training dataset for the emotion recognition task.

I've been only doing 1 and 3. I thought maybe by adding 2, which is what I did
here, it will help the optimizer to learn better, but it didn't.

Probably the step 2 has resulted in a worse local optimum point that cannot
be generalized to the downstream tasks. I mean if you consider that the step 1,
which is done by facebook with a huge amount of training data, is what makes
RoBERTa great that can solve downstream tasks, what I did with step 2 lack of
the amount of data.
 
I also thought about augmenting step 2 with next sentence prediction task (NSP),
but I'm sure this won't work either. If I pretrain on my MELD training data,
it won't do any good to solve the downstream emotion recognition task.

What would be very interesting to try is to incorporate MLM loss and NSP loss
in step 3 and remove step 2. Well, MLM is already incorporated in step 1, so 
it'd be interesting to add NSP loss in the downstream emotion recognition task.
The reason why I want to incldue the NSP loss is that in our case, I assume 
that it is very important to do sequence of sequence modeling for the current 
speaker emotion recognition task, than other tasks. However to achieve this,
I have to hack the code.