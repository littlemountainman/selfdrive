#!/bin/bash

wget --continue --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B-KJCaaF7elleG1RbzVPZWV4Tlk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B-KJCaaF7elleG1RbzVPZWV4Tlk" -O driving_dataset.zip && rm -rf /tmp/cookies.txt

unzip driving_dataset.zip
