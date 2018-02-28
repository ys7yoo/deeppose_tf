import os

# Full path to the project root
ROOT_DIR = os.path.expanduser('~/src/deeppose')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'out')

# put data in /tmp
LSP_DATASET_ROOT = os.path.expanduser('/var/data/lsp')
LSP_EXT_DATASET_ROOT = os.path.expanduser('/var/data/lsp_ext')
MPII_DATASET_ROOT = os.path.expanduser('/var/data/mpii')

#default location (old)
#LSP_DATASET_ROOT = os.path.expanduser('~/data/lsp')
#LSP_EXT_DATASET_ROOT = os.path.expanduser('~/data/lsp_ext')
#MPII_DATASET_ROOT = os.path.expanduser('~/data/mpii')
