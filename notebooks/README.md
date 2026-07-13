# This folder should be empty

This folder should be empty, it is for storing a guide how to do use git (github) in google colab

## Core code

Don't forget to change your name.

```
import os
from google.colab import userdata

# Retrieve credentials
token = userdata.get('GITHUB_TOKEN')
username = "QuantumTacoNinja" # YOUR GITHUB NAME
repo_name = "what-you-eat-decides-your-grade"

# Configure git URL
repo_url = f"https://{token}@github.com/{username}/{repo_name}.git"

# Clone if it doesn't exist, otherwise pull latest
if not os.path.exists(repo_name):
    !git clone {repo_url}
    %cd {repo_name}
else:
    %cd {repo_name}
    !git pull

```

## Commit 

Commit using File -> Save to Github, notebook will commit and save at `./`


## Setup

### colab key

Go to left bar, there should be a key icon, add a key called `GITHUB_TOKEN`

### github token

Go to setting -> developer settings -> personal access token -> generate new token -> clasic -> CHECK repo -> copy token to colab key.


## Save model

Using magic of torch

IDK how it work but model should be save at `./models/`
