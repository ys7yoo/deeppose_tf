#! /bin/bash
# Script for downloading Datasets: LSP, LSP Extended

# Get LSP Dataset
#wget http://www.comp.leeds.ac.uk/mat4saj/lsp_dataset_original.zip
#unzip lsp_dataset_original.zip
#unzip lsp_dataset_original.zip
#rm -rf lsp_dataset_original.zip
# ADDRESS CHANGED
wget http://sam.johnson.io/research/lsp_dataset.zip
unzip lsp_dataset.zip
unzip lsp_dataset.zip
rm -rf lsp_dataset.zip

mkdir lsp
mv images lsp/
mv joints.mat lsp/
mv README.txt lsp/

# move it to ~/data
mv lsp ~/data/


# Get LSP Extended Training Dataset
#wget http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip
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
