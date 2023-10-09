from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
from dotenv import load_dotenv

load_dotenv()

account_url = (os.environ["AZ_STORAGE_ENDPOINT"])
credential = (os.environ["AZ_STORAGE_KEY"])

# Initialize the BlobServiceClient
blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

# Get the ContainerClient
container_client = blob_service_client.get_container_client("flickr30k")

# Directory containing the files you want to upload
local_directory = "C:/Users/tutin/Downloads/archive/flickr30k_images/flickr30k_images"

# Iterate through the local files
for filename in os.listdir(local_directory):
    filepath = os.path.join(local_directory, filename)
    
    # Create a blob client
    blob_client = container_client.get_blob_client(filename)
    
    # Upload the local file
    with open(filepath, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"File '{filename}' uploaded.")
print("All files uploaded!")
