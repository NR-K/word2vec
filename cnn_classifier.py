import tensorflow_datasets as tfds
import tnesorflow as tf

dataset, info = tfds.load("imdb_reviews/subwords8k", with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']