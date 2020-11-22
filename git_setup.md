# Setting up git configurations:

## Setting up user email and name configuration for git:

git config --global user.email "david.raymond.kearney@gmail.com"
git config --global user.name "David Kearney"


## Setting Up SSH Keys
ssh-keygen -t ed25519 -C "david.raymond.kearney@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
