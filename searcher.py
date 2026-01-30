from PIL import Image
import torch
import torch.nn.functional as F
import glob, os
import clip
import numpy as np

def load_image_paths(path):
    return glob.glob(os.path.join(path, '*.[jJ][pP][gG]')) + glob.glob(os.path.join(path, '*.[pP][nN][gG]'))

class EmbeddedSearcher():
    def __init__(self, path):
        self.path = path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)

        self.paths = load_image_paths(self.path)

        images = [self.preprocess(Image.open(p)).unsqueeze(0) for p in self.paths]
        image_input = torch.cat(images).to(self.device)

        with torch.no_grad():
            self.image_features = self.model.encode_image(image_input)
            self.image_features /= self.image_features.norm(dim=-1, keepdim=True)

    def query(self, text, k=1):
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
        
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                logit_scale = self.model.logit_scale.exp()
                logits_per_text = logit_scale * text_features @ self.image_features.t()

                probs = logits_per_text.softmax(dim=-1).cpu().numpy()[0]

            data = [ ( i, self.paths[i], p ) for i, p in enumerate( probs.tolist() )]

            data.sort(reverse=True, key=(lambda x: (x[2])))

            return data[0:min(k,len(data))]

        
        except Exception as e:
            print(f"Error: {e}")
            return []