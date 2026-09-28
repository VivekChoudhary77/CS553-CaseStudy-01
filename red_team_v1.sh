#!/bin/bash

# Number possible groups
num_files=25

# vars
PORT=22000
MACHINE=paffenroth-23.dyn.wpi.edu
#TODO Update file name
KEY=$HOME/projects/1_classes/DS553_private/scripts/CS2/student-admin_key
chmod 600 ${KEY}
WEBHOOK_URL="https://discord.com/api/webhooks/1554256448081502314/muodhTKDVj4Lp8Rv4pSy36bwV0OItbReH9fdiO8303oVbIqBtIO8fBbIicmAH9niZZJ0"

# Loop to create files
for i in $(seq 1 $num_files); do
  echo "trying group ${i} at port $((${i} + ${PORT})) "
  echo "------------------------------------------------"
  echo "------------------------------------------------"
  if ssh -i $KEY -p $((${i} + ${PORT})) -o StrictHostKeyChecking=no student-admin@${MACHINE} hostname; then
    echo "group ${i} is vulnerable!"
    
    #send message to discord
    #code was edited by ChatGPT, added a content field to the JSON, promt was "will this work?"
    curl -H "Content-Type: application/json" -X POST -d '{"content":"successfully accessed another teams machine"}' ${WEBHOOK_URL}
    
    #log onto vulnerable machine
    ssh -i ${KEY} -p $((${i} + ${PORT})) student-admin@${MACHINE}
    exit 0
    
  else
    echo "group ${i} is protected!"
  fi
  echo "------------------------------------------------"
  echo "------------------------------------------------"
done
