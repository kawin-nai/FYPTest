from azure.storage.fileshare import ShareServiceClient, ShareClient, ShareDirectoryClient, ShareFileClient

# Connect to azure file share (filelisttest, access token)
conn_string = "DefaultEndpointsProtocol=https;AccountName=filelisttest;AccountKey" \
              "=cr6aB8V4l1Ogs8XOu36BkVkU1oPhWzDzhbenAuBz71XO/8LFfoUItUFGHA5rfLUkGcHVnpJOrNtP+AStLsXYbg" \
              "==;EndpointSuffix=core.windows.net "
service = ShareServiceClient(account_url="https://filelisttest.file.core.windows.net/", credential="cr6aB8V4l1Ogs8XOu36BkVkU1oPhWzDzhbenAuBz71XO/8LFfoUItUFGHA5rfLUkGcHVnpJOrNtP+AStLsXYbg==")
file_client = ShareFileClient.from_connection_string(conn_str=conn_string, share_name="filelisttest", file_path="application-data")
with open("test.py", "rb") as data:
    file_client.upload_file(data)
print("Done")