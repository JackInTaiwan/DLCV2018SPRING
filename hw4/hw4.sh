if ! [ -d $2 ]; then
    mkdir $2
fi
wget https://www.dropbox.com/s/mite6luwo2agrkq/ave_model_14.pkl?dl=0 -O ave_model_14.pkl
wget https://www.dropbox.com/s/9569a8mahwobeax/gan_gn_fin.pkl?dl=0 -O gan_gn_fin.pkl
wget https://www.dropbox.com/s/v3us5v8ly8ypfpw/acgan_gn_fin.pkl?dl=0 -O acgan_gn_fin.pkl
mv ave_model_14.pkl ./VAE/models/
mv gan_gn_fin.pkl ./GAN/models/
mv acgan_gn_fin.pkl ./ACGAN/models/

echo 'Problem1...'
python3 ./VAE/problems.py -q lcurve --output $2 --record ./VAE/record/vae_fin.json
python3 ./VAE/problems.py -q test --output $2 --model ./VAE/models/ave_model_14.pkl --dataset $1
python3 ./VAE/problems.py -q rg --output $2 --model ./VAE/models/ave_model_14.pkl
python3 ./VAE/problems.py -q tsne --output $2 --model ./VAE/models/ave_model_14.pkl --dataset $1

echo 'Problem1...'
python3 ./GAN/problems.py -q lcurve --output $2 --record ./GAN/record/gan_fin.json
python3 ./GAN/problems.py -q rg --output $2 --model ./GAN/models/gan_gn_fin.pkl

echo 'Problem1...'
python3 ./ACGAN/problems.py -q lcurve --output $2 --record ./ACGAN/record/acgan_fin.json
python3 ./ACGAN/problems.py -q rg --output $2 --model ./ACGAN/models/acgan_gn_fin.pkl

