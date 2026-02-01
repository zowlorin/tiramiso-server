
from PIL import Image
import torch
import torch.nn.functional as F
import glob, os
import clip
import numpy as np

def load_image_paths(path):
    return glob.glob(os.path.join(path, '*.[jJ][pP][gG]')) + glob.glob(os.path.join(path, '*.[pP][nN][gG]')) + glob.glob(os.path.join(path, '*.[jJ][pP][eE][gG]'))

ENGINEERED_TEMPLATES = [
    "a photo of a {}",
    "a picture of a {}",
    "a close-up photo of a {}",
    "a high quality photo of a {}",
]

class EmbeddedSearcher():
    def __init__(self, path):
        self.path = path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        self.paths = load_image_paths(self.path)
        if (not self.paths):
            self.image_features = torch.empty((0, self.model.visual.output_dim), device=self.device)
            return

        images = [self.preprocess(Image.open(p)).unsqueeze(0) for p in self.paths]
        image_input = torch.cat(images).to(self.device)

        with torch.no_grad():
            self.image_features = self.model.encode_image(image_input)
            self.image_features /= self.image_features.norm(dim=-1, keepdim=True)

    def update(self):
        new_paths = load_image_paths(self.path)

        added = [p for p in new_paths if p not in self.paths]
        removed = [p for p in self.paths if p not in new_paths]

        for removed_path in removed:
            index = self.paths.index(removed_path)
            self.paths.pop(index)

            self.image_features = torch.cat([
                self.image_features[:index],
                self.image_features[index+1:]
            ], dim=0)

        for added_path in added:
            self.paths.append(added_path)

            image = self.preprocess(Image.open(added_path).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(image)
                feat /= feat.norm(dim=-1,keepdim=True)

            self.image_features = torch.cat([self.image_features, feat], dim=0)

    def add(self, path):
        if not os.path.exists(path): return None
        image = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model.encode_image(image)
            feat /= feat.norm(dim=-1,keepdim=True)

        self.image_features = torch.cat([self.image_features, feat], dim=0)
        self.paths.append(path)
        return len(self.paths)-1

    def query(self, text, start=0, count=1):
        try:
            texts = [t.format(text) for t in ENGINEERED_TEMPLATES]
            text_tokens = clip.tokenize(texts).to(self.device)
        
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, dim=-1)

                text_features = text_features.mean(dim=0, keepdim=True)
                text_features = F.normalize(text_features, dim=-1)

                logit_scale = self.model.logit_scale.exp()
                logits_per_text = logit_scale * text_features @ self.image_features.t()

                probs = logits_per_text.softmax(dim=-1).cpu().numpy()[0]

            data = [ (self.paths[i].replace(os.sep, "/"), p ) for i, p in enumerate( probs.tolist() )]

            data.sort(reverse=True, key=(lambda x: (x[1])))

            return data[start : start + count]

        
        except Exception as e:
            print(f"Error: {e}")
            return []