#!/bin/bash

cd ..
rm -rf ./aps
rm -f ./aps*

# Application Performance Snapshot launches the application and runs the data collection.
aps ./bin/Engine
