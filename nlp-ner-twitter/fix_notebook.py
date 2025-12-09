"""
Script to add more aggressive TensorFlow warning suppression
"""
import json

# Load the notebook
with open('NER_Twitter_Analysis.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Enhanced imports with full TensorFlow warning suppression
new_imports_source = [
    "# Suppress all TensorFlow warnings BEFORE importing TensorFlow\n",
    "import os\n",
    "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only\n",
    "os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "warnings.filterwarnings('ignore', category=DeprecationWarning)\n",
    "warnings.filterwarnings('ignore', category=FutureWarning)\n",
    "\n",
    "import logging\n",
    "logging.getLogger('tensorflow').setLevel(logging.ERROR)\n",
    "logging.getLogger('transformers').setLevel(logging.ERROR)\n",
    "\n",
    "# Suppress TensorFlow AutoGraph warnings\n",
    "import tensorflow as tf\n",
    "tf.get_logger().setLevel('ERROR')\n",
    "tf.autograph.set_verbosity(0)\n",
    "\n",
    "# Suppress deprecation warnings from TensorFlow\n",
    "import tensorflow.python.util.deprecation as deprecation\n",
    "deprecation._PRINT_DEPRECATION_WARNINGS = False\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from collections import Counter\n",
    "\n",
    "print(\"Libraries imported successfully!\")"
]

fixes_applied = 0

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Find and update the imports cell
        if 'Libraries imported successfully' in source and 'import numpy as np' in source:
            cell['source'] = new_imports_source
            cell['outputs'] = []  # Clear outputs
            fixes_applied += 1
            print("✓ Enhanced TensorFlow warning suppression applied")

# Save the updated notebook
with open('NER_Twitter_Analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"\nTotal fixes applied: {fixes_applied}")
print("Notebook saved successfully!")
