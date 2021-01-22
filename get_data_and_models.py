import os
import gdown

# Make data directory
if not os.path.exists('./data'):
  os.makedirs('./data')
if not os.path.exists('./models'):
  os.makedirs('./models')

# Download ISBI2015 datasets and pretrained models
urls = [
  'https://drive.google.com/uc?id=1eDIYn_cXtPy8RpR16sNpDM4murmvVa69',
  'https://drive.google.com/uc?id=1Z0RwhkGQYIOZuypI9q-Vqj6dE8WuJDaG'
]
outputs = ['data.tgz', 'models.tgz']

for url, output in zip(urls, outputs):
  gdown.download(url, output, quiet=False)

# Extract tgz files
os.system('tar -xvf data.tgz')
os.system('tar -xvf models.tgz')