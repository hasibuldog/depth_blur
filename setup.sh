
set -e

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

if ! command_exists python3; then
    echo "Python is not installed. Please install Python 3."
    exit 1
fi

if ! command_exists pip3; then
    echo "pip is not installed. Please install pip."
    exit 1
fi

if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found!"
    exit 1
fi

echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

REPO_URL="https://github.com/xinntao/Real-ESRGAN.git"
REPO_DIR="Real-ESRGAN"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning Real-ESRGAN repository..."
    git clone $REPO_URL
else
    echo "Real-ESRGAN repository already cloned."
fi

cd $REPO_DIR
echo "Setting up Real-ESRGAN..."
python setup.py develop

cd ..

echo "Setup complete."