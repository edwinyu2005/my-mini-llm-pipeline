#!/bin/bash

# 1. Create the virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install external dependencies
pip install -r requirements.txt

# 5. Install the current project in editable mode
# This makes 'src' importable throughout the project
pip install -e .

echo "Setup complete! Project 'my-mini-llm-pipeline' is now installed in editable mode."
