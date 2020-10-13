#%%
import tensorflow_datasets as tfds
import tensorflow as tf

# %%
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

#%%
train_data, test_data = imdb['train'], imdb['test']
# %%
training_sentences = []
testing_sentences = []

training_labels = []
testing_labels = []
# %%
for s,l in train_data:
  training_sentences.append(s.numpy().decode('utf8'))
  training_labels.append(l.numpy())

# %%
for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())
# %%
import numpy as np
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

#%%
def preprocess_text(text, vocab_size = 10000, embedding_dim = 16, max_len = 150, trunc_type = 'post', oov_tok = '<OOV>'):
    sequence = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequence, maxlen=max_len, truncating=trunc_type)
    return padded

# %%
vocab_size = 10000
embedding_dim = 16
max_len = 150
trunc_type = 'post'
oov_tok = '<OOV>'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequence = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequence, maxlen=max_len, truncating=trunc_type)

testing_sequence = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequence, maxlen=max_len)

# %%
reverse_word_index = {value: key for (key, value) in word_index.items()}
# %%
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[3]))
print(training_sentences[3])

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    # tf.keras.layers.Flatten(),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# %%
history = model.fit(padded, training_labels_final, epochs=10, validation_data=(testing_padded, testing_labels_final))

#%%
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

#%%
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

#%%
model.save("imdb.h5")

# %%
review1 = ["The darker side of superheroes is an area that’s been explored before but not with nearly the intensity and thoughtfulness you’ll see in “The Boys.” An action-packed drama with touches of humor, this is a TV series you won’t want to miss."]

# %%
review1 = preprocess_text(review1)

# %%
model.predict(review1)
# %%
review2 =["Maybe some day Mahesh Bhatt will make something watchable with Alia Bhatt, one of the most exciting actors of this generation. Sadly, Sadak 2 is not that film."]
review2 = preprocess_text(review2)
model.predict(review2)
# %%
