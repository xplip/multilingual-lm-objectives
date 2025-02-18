#!/bin/bash

export UD_URL="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3687/ud-treebanks-v2.8.tgz"
export UD_ARCHIVE=$(basename $UD_URL)

# Download
wget "$UD_URL"

# Unpack
tar -xvzf "$UD_ARCHIVE"

# Clean up
rm "$UD_ARCHIVE"
