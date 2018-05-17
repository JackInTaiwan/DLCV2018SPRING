python3 ./VAE/problems.py -q tsne --output $2 --model ./VAE/models/ave_model_14.pkl --dataset $1
python3 ./VAE/problems.py -q lcurve --output $2 --record ./VAE/record/vae_fin.json
python3 ./VAE/problems.py -q rg --output $2 --model ./VAE/models/ave_model_14.pkl
python3 ./VAE/problems.py -q test --output $2 --model ./VAE/models/ave_model_14.pkl --dataset $1

python3 ./GAN/problems.py -q lcurve --output $2 --record ./GAN/record/gan_fin.json
python3 ./GAN/problems.py -q rg --output $2 --model ./GAN/models/gan_gn_fin.pkl

