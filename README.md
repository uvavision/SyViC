# SyViC
This repository is an official implementation of the paper "Going beyond nouns with vision & language models using synthetic data" (ICCV 2023). Here, we present two contributions: the Synthetic Visual Concepts (SyViC) dataset and a novel training strategy for Vision & Language (VL) models. Together, these enhance the models' understanding of complex language concepts without compromising zero-shot capabilities. 

# Dataset
Our million-scale dataset is available on Google Drive for download: [https://drive.google.com/drive/folders/1GiS1smsSg3NwwwZ4xQqAuLCvS7vAawWs](https://drive.google.com/drive/folders/1GiS1smsSg3NwwwZ4xQqAuLCvS7vAawWs).

### Google Drive Rate Limitation
**Temporary Issue**: Some users may encounter a rate limit on access requests to our dataset. In particular, you may encounter the following message:

```
Sorry, you can't view or download this file at this time.

Too many users have viewed or downloaded this file recently. Please try accessing the file again later. If the file you are trying to access is particularly large or is shared with many people, it may take up to 24 hours to be able to view or download the file. If you still can't access a file after 24 hours, contact your domain administrator.
```

**What to do if you encounter this?**

1. Copy to Your Drive:
	* Open the dataset link.
	* Locate the Copy to my Drive or Add to My Drive option (usually represented by a Drive icon).
	* Once copied, you can download the dataset directly from your own Drive, avoiding the public access rate limit. Remember, the dataset may occupy significant space (worth GBs) on your Drive, so ensure you have the necessary storage available.
	* Try Again Later: If you prefer not to copy it to your Drive or face any other issues, kindly wait for a while and try accessing the dataset again later. The rate limits usually reset after a certain period.

We apologize for the inconvenience and are working on finding a more scalable hosting solution. Thank you for your understanding and patience.

# Data Generation
We provide code and instructions for dataset generation and extensibility in `./dataset_generation`.

# Finetuning
### **Setup**:
Create a virtual environment with required dependancies.
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r finetuning/requirements.txt
```

### Dataset Preprocessing
[Download our dataset from Google Drive](https://drive.google.com/drive/u/2/folders/1GiS1smsSg3NwwwZ4xQqAuLCvS7vAawWs) and place it in `finetuning/datasets/syvic`.
* You may alternatively generate a new dataset using the dataset generation codebase.
* If you would like to use additional training features such as AdaIN and MixStyle, you will need to download the Human Motion Database HMDB51 dataset and the Imagenet dataset and place them in the `finetuning/datasets/hmdb51` and `finetuning/datasets/imagenet` folders, respectively.


### Training
Run `finetuning/main.py` with the appropriate flags . For example, the following command trains CLIP using rank-16 LoRa adapters and using caption splitting without using domain adaptation:

```bash
cd finetuning

python main.py  --seed 34 --capsplit --steps 10 --backbone_name ViT-B/32 --folder_name results/synCLIP_lora_capsplit_/ViT-B32 --lr 0.0025 --weight_decay 0.2 --lora_r 16 --data_type material,size --heavy_aug --early_stop 6 --batch_size 256 --sample_fps 6
```

# Cite Us
	@article{cascante2023going,
	  title={Going beyond nouns with vision \& language models using synthetic data},
	  author={Cascante-Bonilla, Paola and Shehada, Khaled and Smith, James Seale and Doveh, Sivan and Kim, Donghyun and Panda, Rameswar and Varol, Gül and Oliva, Aude and Ordonez, Vicente and Feris, Rogerio and Karlinsky, Leonid},
	  journal={arXiv preprint arXiv:2303.17590},
	  year={2023}
	}
