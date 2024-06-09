
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

if ! command_exists git; then
    echo "git is not installed. Please install git."
    exit 1
fi

# Add the Real-ESRGAN submodule
echo "Adding Real-ESRGAN submodule..."
git submodule add https://github.com/xinntao/Real-ESRGAN.git Real-ESRGAN

cd Real-ESRGAN
echo "Setting up Real-ESRGAN..."
python setup.py develop

cd ..

echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete."
