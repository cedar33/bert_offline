from __future__ import absolute_import, division, print_function, unicode_literals
from Embeddings_bert import Embedding
import numpy as np
import os
import tensorflow as tf
import ctypes
import tensorflow.compat.v1 as tf1
import matplotlib.pyplot as plt
import collections
import csv
import os
import random
from pathlib import Path
flags = tf1.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("vocab_file", None,
                    "词表文件路径")
flags.DEFINE_string("output_dir", None,
                    "record文件、模型输出路径")
flags.DEFINE_string("data_dir", None,
                    "训练和评估文件路径")
flags.DEFINE_integer("train_batch_size", 32,
                    "batch_size")
flags.DEFINE_integer("num_train_epochs", 5,
                    "epochs")
flags.DEFINE_integer("max_seq_length", 256,
                    "max_seq_length")


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

# hllDll = ctypes.WinDLL(
#     "C:\\Program Files\\NVIDIA Corporation\\NvStreamSrv\\cudart64_100.dll")
# ctypes.WinDLL("D:\\job\\glove_senta\\cublas64_100.dll")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

label_list = ["negative", "positive"]


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


def get_voc_list(filename):
    v_list = []
    with open(filename, "r", encoding="utf-8") as f:
        v_list += f.readlines()
    v_list_2 = [v.replace("\n", "") for v in v_list]
    v_dict = dict()
    for idx, v in enumerate(v_list_2):
        v_dict[v] = idx
    return v_dict


def v_list_lookup(line_list, v_dict):
    """词表查找方法"""
    input_ids = []
    for w in line_list:
        try:
            input_ids.append(v_dict[w])
        except KeyError:
            input_ids.append(v_dict["[unused1]"])
    return input_ids


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 seq_length,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        # self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example
        self.seq_length = seq_length


def convert_single_example(ex_index, example, label_list, max_seq_length):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    v_list = get_voc_list(FLAGS.vocab_file)
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * max_seq_length,
            seq_length=max_seq_length,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = v_list_lookup(example.text, v_list)
    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[0:(max_seq_length)]
    tokens = []
    # segment_ids = []
    # tokens.append("[CLS]")
    # segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        # segment_ids.append(0)
    # tokens.append("[SEP]")
    # segment_ids.append(0)
    input_ids = v_list_lookup(example.text, v_list)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        # segment_ids.append(0)
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[0:max_seq_length]
        input_mask = input_mask[0:max_seq_length]
        # segment_ids = segment_ids[0:max_seq_length]
    seq_length = len(input_ids)
    assert len(input_ids) == max_seq_length, print(len(input_ids))
    assert len(input_mask) == max_seq_length
    # assert len(segment_ids) == max_seq_length
    label_id = [0 for i in range(0, len(label_list))]
    label_id[label_map[example.labels]] = 1
    if ex_index < 5:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
        # print("segment_ids: %s" %
        #                 " ".join([str(x) for x in segment_ids]))
        print("label: %s (id = %s)" % (example.labels, label_id))
        print("seq_length: %s (id = %s)" % (seq_length, label_id))

    feature = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        # "segment_ids": segment_ids,
        "label_ids": label_id,
        "seq_length": seq_length,
        "is_real_example": 1}
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" %
                             (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length)

        def create_int_feature(values):
            if isinstance(values, list):
                f = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=values))
            else:
                f = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[values]))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature["input_ids"])
        features["input_mask"] = create_int_feature(feature["input_mask"])
        # features["segment_ids"] = create_int_feature(feature["segment_ids"])
        features["label_ids"] = create_int_feature(feature["label_ids"])
        features["seq_length"] = create_int_feature([feature["seq_length"]])
        features["is_real_example"] = create_int_feature(
            [int(feature["is_real_example"])])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([len(label_list)], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        _example = tf.io.parse_single_example(record, name_to_features)
        for name in list(_example.keys()):
            t = _example[name]
            # print("dtype : %s" % t.dtype)
            if t.dtype == tf1.int64:
                t = tf1.to_int32(t)
            elif t.dtype == tf1.string:
                t = tf1.to_int32(t)
            _example[name] = t
        example = [_example["input_ids"]], [_example["label_ids"]]
        return example

    def input_fn():
        """The actual input function."""

        d = tf.data.TFRecordDataset(input_file)
        # if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)

        d = d.map(lambda record: _decode_record(record, name_to_features))
        d.batch(
            batch_size=FLAGS.train_batch_size,
            drop_remainder=False
        )

        # d = d.numpy()
        print("dataset shape: %s" % d)

        return d

    return input_fn


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, labels=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf1.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class RecordDataProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_labels(self):
        """See base class."""

        return label_list

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, str(i))
            text = line[0]

            label_t = line[1]

            labels = list()
            if set_type == "test":

                labels = ["contradiction"]
            else:
                labels = label_t
            examples.append(
                InputExample(guid=guid, text=text, labels=labels))
        return examples

# 以上全部是bert中数据预处理方法，有问题可以直接去bert代码中看

def create_model(embeddings_file, vocab_file):
    model = tf.keras.Sequential([
    Embedding(embeddings_file=embeddings_file,
              vocab_file=vocab_file),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(FLAGS.max_seq_length)),
    tf.keras.layers.Dense(FLAGS.max_seq_length, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])
    model.summary()
    return model
if __name__ == "__main__":
    processor = RecordDataProcessor()
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = len(train_examples) / FLAGS.train_batch_size
    # num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)

    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, None, train_file)
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, None, eval_file)
    train_input_fn = file_based_input_fn_builder(
        input_file="model/train.tf_record",
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)()
    eval_input_fn = file_based_input_fn_builder(
        input_file="model/eval.tf_record",
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)()





    # 使用tensorboard查看训练过程
    # tensorboard_log = tf.keras.callbacks.TensorBoard(log_dir="D:\\job\\glove_senta\\model", write_images=1, histogram_freq=1, embeddings_freq=1)
    # log = [tensorboard_log]
    model = create_model(embeddings_file="D:\\job\\glove_senta\\bert_embeddings.npz", vocab_file="D:\\job\\chinese_L-12_H-768_A-12\\chinese_L-12_H-768_A-12\\vocab.txt")
    # 保存模型
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.output_dir,
                                                    save_weights_only=True,
                                                    verbose=1) 
    history = model.fit(x=train_input_fn, epochs=FLAGS.num_train_epochs,
                        validation_data=eval_input_fn,
                        validation_steps=30,
                        callbacks=[cp_callback],
                        steps_per_epoch=1000
                        )

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
