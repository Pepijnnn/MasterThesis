import torch.nn as nn
from torchvision import models


from src.utils import args


# ----- Model selection -----

def load_model(model_name=args.model, n_classes=11):
    # Model selection
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 3)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=False)
        model.classifier = nn.Linear(1024, 14) #3
        # nn.Sigmoid
        # print(model)
        
    elif model_name == 'densenet121_n':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, 50)
        class_layers = []
        for _ in range(n_classes):
            class_layers.append(nn.Linear(50,1))
        class_modules = nn.ModuleList(class_layers).to(args.device)
        model = model.to(args.device)
        return model, class_modules
    elif model_name == 'ownnet':
        model = nn.Sequential(
          nn.Linear(221,300),
          nn.ReLU(),
          nn.Linear(300,300),
          nn.ReLU(),
          nn.Linear(300,150),
          nn.ReLU(),
          nn.Linear(150,75),
          nn.ReLU(),
          nn.Linear(75,50),
          nn.Sigmoid()
        ).to(args.device)
        class_layers = []
        for _ in range(n_classes):
            class_layers.append(nn.Linear(50,1))
        class_modules = nn.ModuleList(class_layers).to(args.device)
        return model, class_modules

    model = nn.DataParallel(model.to(args.device))
    return model


if __name__ == "__main__":
    pass

