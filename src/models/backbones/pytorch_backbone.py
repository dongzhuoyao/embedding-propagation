import torchvision.models as models
import torch



class resnet50(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        resnet50 = models.resnet50(pretrained=False)
        modules = list(resnet50.children())[:-1]
        self._resnet50 = torch.nn.Sequential(*modules)
        self.output_size = 2048

    def add_classifier(self, no, name="classifier", modalities=None):
        setattr(self, name, torch.nn.Linear(self.output_size, no))

    def forward(self, x, *args, **kwargs):
        *dim, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self._resnet50(x)
        return x.view(*dim, self.output_size)

class resnet101(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        resnet = models.resnet101(pretrained=False)
        modules = list(resnet.children())[:-1]
        self._model = torch.nn.Sequential(*modules)
        self.output_size = 2048

    def add_classifier(self, no, name="classifier", modalities=None):
        setattr(self, name, torch.nn.Linear(self.output_size, no))

    def forward(self, x, *args, **kwargs):
        *dim, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self._model(x)
        return x.view(*dim, self.output_size)

class densenet121(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        resnet = models.densenet121(pretrained=False)
        modules = list(resnet.children())[:-1]
        self._model = torch.nn.Sequential(*modules)
        self.output_size = 1024

    def add_classifier(self, no, name="classifier", modalities=None):
        setattr(self, name, torch.nn.Linear(self.output_size, no))

    def forward(self, x, *args, **kwargs):
        *dim, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self._model(x)
        x = torch.mean(x,dim=[2,3])
        return x.view(*dim, self.output_size)

class wrn50_2(torch.nn.Module):
        def __init__(self, dropout):
            super().__init__()
            resnet = models.wide_resnet50_2(pretrained=False)
            modules = list(resnet.children())[:-1]
            self._model = torch.nn.Sequential(*modules)
            self.output_size = 2048

        def add_classifier(self, no, name="classifier", modalities=None):
            setattr(self, name, torch.nn.Linear(self.output_size, no))

        def forward(self, x, *args, **kwargs):
            *dim, c, h, w = x.size()
            x = x.view(-1, c, h, w)
            x = self._model(x)
            x = torch.mean(x, dim=[2, 3])
            return x.view(*dim, self.output_size)
