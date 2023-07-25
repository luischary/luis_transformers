criar chave ssh para poder pegar o repo do git
substituindo o email pelo seu email

ssh-keygen -t ed25519 -C "your_email@example.com"

rodar
eval "$(ssh-agent)"
ssh-add caminho_para_chave_do_git

adicionar a chave do git no git

fazer o git clone via ssh
git clone git@github.com:luischary/luis_transformers.git

# se necessario instala o venv
sudo apt install python3.8-venv

cria o venv
sudo apt install python3.8-venv
python3 -m venv venv


instalar a versao correta do pytorch (atencao para a versao do cuda)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

instalar o requirements
python3 -m pip install -r requirements.txt

instalar o flash-attention separadamente
python3 -m pip install flash-attn --no-build-isolation

# subir dados para a parada via scp
scp -i ~/aws/clone_itau_sp.pem ./tokens_marketplace_512.tar.gz ubuntu@54.233.228.107:/home/ubuntu/luis_transformers/data/tokenized_datasets

scp -i ~/aws/clone_itau_sp.pem ./tokens_marketplace_128.tar.gz ubuntu@18.230.70.85:/home/ubuntu/luis_transformers/data/tokenized_datasets

scp -i ~/aws/clone_itau_sp.pem ./datatokens.tar.gz  ubuntu@18.230.70.85:/home/ubuntu/luis_transformers/data

# unzip das paradas
tar -xf nomedoarquivo

# teste encoder vanilla
python main_train.py --model-name teste_1 --model-path ./modelos_treinados/teste --model-type encoder --dataset-type regular --encoder-type vanilla --training-steps 10000 --batch-size 32

# teste hierarchical com general dataset
python main_train.py --model-name teste_2 --model-path ./modelos_treinados/teste --model-type encoder --dataset-type variable --encoder-type hierarchical --training-steps 1000 --batch-size 1 --dataset-limit 10000

# assistir consumo
watch -n 1 nvidia-smi
