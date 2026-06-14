import random as _random

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

import models.config as cfg


class VQADataset(Dataset):  # Visual Question Answering Dataset
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle image (it's a list)
        image_data = item['images']
        if isinstance(image_data, list) and len(image_data) > 0:
            image = image_data[0]
        else:
            image = image_data

        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(
                3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)

        # Process text (also a list). Cauldron packs ~several QA pairs per image;
        # sample one per visit so repeated epochs see different text targets.
        text_data = item['texts']
        if isinstance(text_data, list) and len(text_data) > 0:
            text = _random.choice(text_data)
        else:
            text = text_data

        question = text['user']
        # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        answer = text['assistant'] + self.tokenizer.eos_token

        formatted_text = f"Question: {question} Answer:"

        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }


class IterableVQADataset(IterableDataset):
    """Wraps an HF IterableDataset, applying the same image+text transform as VQADataset.

    Cloud dry-run seam: DataLoader(num_workers=0) required locally; multi-worker sharding
    (IterableDataset.shard per worker_info) must be validated on the GPU instance.
    """
    def __init__(self, hf_iterable, tokenizer, image_processor):
        self.hf_iterable = hf_iterable
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __iter__(self):
        # workers>0 OK: HF IterableDataset auto-distributes its data-source shards
        # across DataLoader workers (disjoint, no dup/loss) when iterated in a worker.
        # mix exposes n_shards=41 -> supports up to 41 workers. closes the dry-run seam.
        for item in self.hf_iterable:
            image_data = item['images']
            image = image_data[0] if isinstance(image_data, list) and image_data else image_data
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                processed_image = self.image_processor(image)
            else:
                processed_image = torch.zeros(
                    3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)

            text_data = item['texts']
            text = _random.choice(text_data) if isinstance(text_data, list) and text_data else text_data
            question = text['user']
            answer = text['assistant'] + self.tokenizer.eos_token
            yield {
                'image': processed_image,
                'text_data': f'Question: {question} Answer:',
                'answer': answer,
            }


class MMStarDataset(Dataset):  # https://huggingface.co/datasets/Lin-Chen/MMStar
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        image = item['image']
            
        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {idx}")
            # Create empty tensor with right dimensions as fallback
            processed_image = torch.zeros(3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size)
        
        question = item['question']
        answer = item['answer'] + self.tokenizer.eos_token # Add EOS token to the answer to train model to predict it, enabling correct stopping during generation
        
        formatted_text = f"Question: {question} \nAnswer only with the letter! \nAnswer:"
        
        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer
        }
    