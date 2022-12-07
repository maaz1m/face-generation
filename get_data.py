import os
import gdown
from zipfile import ZipFile

data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

input_url = 'https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684'
file_name = 'img_align_celeba.zip'
file_path = os.path.join(data_dir, file_name)
gdown.download(input_url, file_path)

print('Unzipping...')
with ZipFile(file_path, 'r') as zipobj:
    zipobj.extractall(data_dir)
os.remove(file_path)
print(f'Data downloaded to {os.path.splitext(file_path)[0]}')
