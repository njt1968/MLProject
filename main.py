from dotenv import load_dotenv
import os
from azure.storage.blob import BlobServiceClient
import io
from PIL import Image
import pandas as pd

load_dotenv()

account_url = (os.environ["AZ_STORAGE_KEY"])
credential = (os.environ["AZ_STORAGE_ENDPOINT"])


# Initialize the BlobServiceClient
blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

# Get container client
container_client = blob_service_client.get_container_client("flickr30k")

# Get blob client for the 'results.csv'
blob_client_csv = container_client.get_blob_client("results.csv")

# Download the blob data into a stream
stream_csv = blob_client_csv.download_blob()

df = pd.read_csv(io.StringIO(stream_csv.readall().decode('utf-8')), sep='|')



# Get all blobs in the container
blob_list = container_client.list_blobs()

# Loop through each blob
for blob in blob_list:
    blob_client = container_client.get_blob_client(blob.name)
    
    # Download the blob data into a stream
    stream = blob_client.download_blob()
    
    # Convert blob stream to image (using PIL)
    image = Image.open(io.BytesIO(stream.readall()))
