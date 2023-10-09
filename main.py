from dotenv import load_dotenv
import os
import io
import math
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from azure.storage.blob import BlobServiceClient
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Embedding, LSTM, Add, Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
account_url = (os.environ["AZ_STORAGE_ENDPOINT"])
credential = (os.environ["AZ_STORAGE_KEY"])


def main():
    print("Starting main function.")
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    container_client = blob_service_client.get_container_client("flickr30k")

    print("Loading labels...")
    df = load_labels(container_client)
    print(f"Loaded dataframe from Azure. Length of {len(df)}.")    
    print("DataFrame columns:", df.columns)

    print("Splitting data into train and test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.dropna(subset=['comment'], inplace=True)
    test_df.dropna(subset=['comment'], inplace=True)

    print(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")

    print("Creating data generators...")
    train_gen = AzureBlobDataGenerator(train_df, container_client, batch_size=32)
    test_gen = AzureBlobDataGenerator(test_df, container_client, batch_size=32)
    
    print("Building CNN model...")
    input_shape = (224, 224, 3)
    cnn_model = build_CNN_model(input_shape)
    print("Compiling CNN model...")
    cnn_model.compile(optimizer='adam', loss='mean_squared_error') 

    print("Testing data generator output types and shapes...")
    X_test, y_test = next(iter(train_gen))
    print(f"Type of X_test: {type(X_test)}, Shape of X_test: {X_test.shape}")
    print(f"Type of y_test: {type(y_test)}, Shape of y_test: {y_test.shape}")
    print(f"Type of y_test[0]: {type(y_test[0])}")

    print("Fitting CNN model...")
    cnn_model.fit(train_gen, epochs=10, validation_data=test_gen)

    print("Setting up parameters for combined model...")
    vocab_size = 5000
    embedding_dim = 256
    rnn_units = 512
    max_caption_length = 20

    print("Building and compiling combined model...")
    model = build_combined_model(input_shape, vocab_size, embedding_dim, rnn_units, max_caption_length)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print("Main function completed.")
    

def load_labels(container_client):
    blob_client_csv = container_client.get_blob_client("results.csv")

    stream_csv = blob_client_csv.download_blob()

    df = pd.read_csv(io.StringIO(stream_csv.readall().decode('utf-8')), sep='|')
    
    df.columns = df.columns.str.strip()
    return df


class AzureBlobDataGenerator(Sequence):
    def __init__(self, df, container_client, batch_size=32):
        self.df = df
        self.container_client = container_client
        self.batch_size = batch_size
        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(self.df['comment'].values) 

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, idx):
        batch_df = self.df[idx * self.batch_size: (idx + 1) * self.batch_size]
        X_batch = []
        y_batch = []

        for _, row in batch_df.iterrows():
            blob_name = row['image_name']
            blob_client = self.container_client.get_blob_client(blob_name)
            stream = blob_client.download_blob()

            # Convert blob stream to image
            image = Image.open(io.BytesIO(stream.readall())).resize((224, 224))
            image_array = np.array(image)

            X_batch.append(image_array)
            y_batch.append(row['comment'])

        # Tokenize y_batch
        y_batch_tokenized = self.tokenizer.texts_to_sequences(y_batch)
        y_batch_padded = pad_sequences(y_batch_tokenized, padding='post')
        return np.array(X_batch), np.array(y_batch_padded)




def build_CNN_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))  # Change this based on your specific use case
    return model

def build_CNN_feature_extractor(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    return model

def build_RNN_model(vocab_size, embedding_dim, input_length, rnn_units):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(LSTM(rnn_units, return_sequences=True))
    model.add(Dense(vocab_size, activation='softmax'))
    return model

def build_combined_model(input_image_shape, vocab_size, embedding_dim, rnn_units, max_caption_length):
    # CNN part
    image_input = Input(shape=input_image_shape)
    cnn_model = build_CNN_feature_extractor(input_image_shape)
    image_features = cnn_model(image_input)

    # RNN part
    caption_input = Input(shape=(max_caption_length,))
    rnn_model = build_RNN_model(vocab_size, embedding_dim, max_caption_length, rnn_units)
    caption_embedding = rnn_model(caption_input)
    
    # Combine CNN and RNN
    decoder_input = Add()([image_features, caption_embedding])
    output = LSTM(rnn_units, return_sequences=True)(decoder_input)
    output = Dense(vocab_size, activation='softmax')(output)

    # Combined model
    model = Model(inputs=[image_input, caption_input], outputs=output)
    return model


if __name__ == "__main__":
    main()
