### BERT offline
使用BERT的输出作为词向量，下游任务中不再对词向量进行训练，效果优于一般词向量模型，又显著的减少了显存占用和推理时间，基于此词向量的双向lstm的文本二分类任务在4g显存的笔记本上能以300 examples/s 的速度推，验证集准确率93%左右。  
`bert.npz`: BERT的输出，做为下游任务的`embedding`层  
`vocab.txt`: BERT词表  
`text_classify.py`: 文本分类任务样例，基于`tensorflow2.0`，运行脚本：

```Bash
python text_classify.py --data_dir=./ --output_dir=./model/ --vocab_file=./vocab.txt --train_batch_size=32 --num_train_epochs=10 --max_seq_length=256python text_classify.py --data_dir=./ --output_dir=./model/ --vocab_file=./vocab.txt --train_batch_size=32 --num_train_epochs=10 --max_seq_length=256
```  

`tran.tsv, dev_matched.tsv`: 训练、测试数据，格式：
```plaintext
我从山中来，带着兰花草  negative
```
以`\t`为分隔符  
ps: 词表查找通过`list`的`index`方法查找的，效率比较低，对速度要求比较高的可以修改成字典方式，或者修改bert自带的`tokenizer`来实现词表高速查找

***
BERT offline is a simple but efficient way to use BERT embeddings output on some downstream task such as text classification and sequence labeling. It dumps bert's last output layer as a numpy array and will not be trained during downstream's training. the  accuracy of BERT offline on sst-2 is about 93% on validition dataset and performs well on `1050ti(4g)` GPU, about 300 examples/s during inference.  
`bert.npz`: output of BERT, can be embedding layer on downstream task  
`vocab.txt`: bert's vocabulary list  
`text_classify.py`: an example code of text calssification，based on `tensorflow2.0`：
```Bash
python text_classify.py --data_dir=./ --output_dir=./model/ --vocab_file=./vocab.txt --train_batch_size=32 --num_train_epochs=10 --max_seq_length=256python text_classify.py --data_dir=./ --output_dir=./model/ --vocab_file=./vocab.txt --train_batch_size=32 --num_train_epochs=10 --max_seq_length=256
```  
`tran.tsv, dev_matched.tsv`: training,validation dataset，format：
```plaintext
I am groot  negative
```
delimit `\t`  
ps: I use `list.index()` method to find input_ids for text, you can use a `dict` or modify bert's `tokenizer` to speed up.

