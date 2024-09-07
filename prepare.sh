apt-get install python3-dev g++ libgl1 git-lfs
git lfs install
git submodule update --init --recursive

pip install -r requirements.txt
touch inswapper/__init__.py
pip install -r inswapper/requirements.txt