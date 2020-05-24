import tensorflow as tf
import os

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.model_selection import train_test_split

from transformers import TFBertForSequenceClassification, BertTokenizer, BertConfig, TFBertForTokenClassification, BertForMaskedLM

from tqdm import tqdm


def check_gpu_memory():
    print(f"Tensorflow version: {tf.__version__}")

    # Restrict TensorFlow to only allocate 4GBs of memory on the first GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"The system contains '{len(gpus)}' Physical GPUs and '{len(logical_gpus)}' Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print(f"Your system does not contain a GPU that could be used by Tensorflow!")


def train_model():
    bert_config = BertConfig.from_pretrained("pretrained_models/slo-hr-en-bert-pytorch")
    bert_model = TFBertForSequenceClassification.from_pretrained("pretrained_models/slo-hr-en-bert-pytorch",
                                                                 config=bert_config, from_pt=True)

    def convert_to_input(row):
        pad_token = 0
        pad_token_segment_id = 0
        max_length = 128

        input_ids, attention_masks, token_type_ids = [], [], []

        for x in tqdm(row, position=0, leave=True):
            inputs = bert_tokenizer.encode_plus(x, add_special_tokens=True, max_length=max_length)

            i, t = inputs["input_ids"], inputs["token_type_ids"]
            m = [1] * len(i)

            padding_length = max_length - len(i)

            i = i + ([pad_token] * padding_length)
            m = m + ([0] * padding_length)
            t = t + ([pad_token_segment_id] * padding_length)

            input_ids.append(i)
            attention_masks.append(m)
            token_type_ids.append(t)

        return [np.asarray(input_ids),
                np.asarray(attention_masks),
                np.asarray(token_type_ids)]

    def convert_to_dataset(input_ids, attention_masks, token_type_ids, y):
        return {"input_ids": input_ids,
                "attention_mask": attention_masks,
                "token_type_ids": token_type_ids}, y

    data = pd.read_csv('cache/baseline.csv')

    # Transform negative to 0, neutral to 1 and positive to 2
    label_encoder = preprocessing.LabelEncoder()
    data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

    X = (np.array(data['context']))
    y = (np.array(data['sentiment']))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    print("Train dataset shape: {0}, \nTest dataset shape: {1} \nValidation dataset shape: {2}".format(X_train.shape,
                                                                                                       X_test.shape,
                                                                                                       X_val.shape))

    bert_tokenizer = BertTokenizer.from_pretrained("pretrained_models/slo-hr-en-bert-pytorch", from_pt=True)
    bert_config = BertConfig.from_pretrained("pretrained_models/slo-hr-en-bert-pytorch")
    bert_model = TFBertForSequenceClassification.from_pretrained("pretrained_models/slo-hr-en-bert-pytorch", config=bert_config, from_pt=True)

    X_test_input = convert_to_input(X_test)
    X_train_input = convert_to_input(X_train)
    X_val_input = convert_to_input(X_val)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train_input[0], X_train_input[1], X_train_input[2], y_train)).map(
        convert_to_dataset).shuffle(100).batch(12).repeat(5)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_input[0], X_val_input[1], X_val_input[2], y_val)).map(
        convert_to_dataset).batch(12)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_input[0], X_test_input[1], X_test_input[2], y_test)).map(
        convert_to_dataset).batch(12)

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    bert_history = bert_model.fit(train_ds, epochs=3, validation_data=val_ds)

    bert_model.save_pretrained('pretrained_models/bert-fine-tuned')


if __name__ == '__main__':

    check_gpu_memory()
    train_model()



