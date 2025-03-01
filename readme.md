# COVAS:NEXT AIServer

## Running on Linux

Install pyenv ([Docs](https://github.com/pyenv/pyenv?tab=readme-ov-file#a-getting-pyenv))

```sh
curl -fsSL https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
```

Install pyenv-virtualenv ([Docs](https://github.com/pyenv/pyenv-virtualenv?tab=readme-ov-file#installing-as-a-pyenv-plugin))

```sh
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

Reload your shell

```sh
exec $SHELL
```

Install Python 3.12.8

```sh
pyenv install 3.12.8
```

Create a virtual environment

```sh
cd covas-next-aiserver # if you're not already in this directory
pyenv virtualenv 3.12.8 covas-next-aiserver
pyenv local covas-next-aiserver
```

Install dependencies

```sh
pip install -r requirements.txt
```

Run the server

```sh
python src/AIServer.py
```
