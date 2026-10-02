"""
CNN-LSTM behaviour classifier: ResNet18 (frozen, ImageNet-pretrained) feature
extractor -> LSTM over the 16-frame sequence -> Linear(2) classifier.

Layer names below (`cnn`, `lstm`, `classifier`) match the actual trained
checkpoint exactly, confirmed via inspect_checkpoint.py against the real
state_dict rather than guessed from the summary.

Input:  tensor of shape [B, 16, 3, 224, 224], ImageNet-normalized RGB frames.
Output: logits of shape [B, 2]  (index 0 = Not Smoking, 1 = Smoking)
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SmokingBehaviourModel(nn.Module):
    def __init__(self, hidden_size: int = 256, num_classes: int = 2):
        super().__init__()

        cnn = models.resnet18(weights=None)  # weights loaded from checkpoint below, not ImageNet, at inference
        cnn.fc = nn.Identity()  # matches checkpoint: no cnn.fc.* keys present, just a 512-dim output
        self.cnn = cnn
        for p in self.cnn.parameters():
            p.requires_grad = False  # frozen feature extractor

        # lstm.weight_ih_l0 shape (1024, 512) confirms input_size=512, hidden_size=256 (1024 = 4*256)
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, T, C, H, W]  where T=16
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.cnn(x)                  # [B*T, 512]
        feats = feats.view(b, t, -1)          # [B, T, 512]
        lstm_out, _ = self.lstm(feats)        # [B, T, hidden_size]
        last = lstm_out[:, -1, :]             # final timestep's hidden state
        logits = self.classifier(last)        # [B, 2]
        return logits


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_behaviour_model(checkpoint_path: str, device: str = "cpu") -> SmokingBehaviourModel:
    model = SmokingBehaviourModel()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)  # strict=True by default: will raise loudly on any mismatch
    model.eval()
    model.to(device)
    return model
