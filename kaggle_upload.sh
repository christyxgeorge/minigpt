#!/bin/bash
## Script to upload to Kaggle

DS_NAME="cxgeorge/minigpt"

mkdir -p kaggle/
find . -name '*.py' ! -path './__pycache__/*' ! -path './.history/*' -exec ditto {} kaggle/{} \; 

if kaggle datasets list --mine | grep -q 'cxgeorge/minigpt' ; then
    echo "Dataset already exists, Updating to new version..."

    # Download Metadata file
    kaggle datasets metadata -p ./kaggle cxgeorge/minigpt

    # Update to new version
    kaggle datasets version -p ./kaggle -m "Updated dataset" --dir-mode zip
else
    echo "Creating new dataset..."
    # Download Metadata file template and change title and slug
    kaggle datasets init
    sed -i '' -e "s/INSERT_TITLE_HERE/Minigpt/" -e "s/INSERT_SLUG_HERE/minigpt/" dataset-metadata.json 

    # Create dataset (Keep it private)
    kaggle datasets create -p ./kaggle --dir-mode zip
fi

rm -rf kaggle/
