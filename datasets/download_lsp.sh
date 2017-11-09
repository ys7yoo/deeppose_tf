#! /bin/bash
# Script for downloading Datasets: LSP, LSP Extended

# Get LSP Dataset
# ADDRESS CHANGED
wget http://sam.johnson.io/research/lsp_dataset.zip
unzip lsp_dataset.zip
rm -rf lsp_dataset.zip

mkdir lsp
mv images lsp/
mv joints.mat lsp/
mv README.txt lsp/
mv visualized lsp/

# move it to ~/data
mv lsp ~/data/


# Get LSP Extended Training Dataset
# ADDRESS CHANGED
wget http://sam.johnson.io/research/lspet_dataset.zip
unzip lspet_dataset.zip
rm -rf lspet_dataset.zip

mkdir lsp_ext
mv images lsp_ext/
mv joints.mat lsp_ext/
mv README.txt lsp_ext/

# move it to ~/data
mv lsp_ext ~/data/
