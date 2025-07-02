
import torch
import torch.nn as nn
import timm

class DINOModel(nn.Module):
    def __init__(self, backbone_name="vit_small_patch16_224", out_dim=256, momentum=0.996):
        super().__init__()
        self.student = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.teacher = timm.create_model(backbone_name, pretrained=False, num_classes=0)

        self.student_projector = nn.Sequential(
            nn.Linear(self.student.num_features, 2048),
            nn.GELU(),
            nn.Linear(2048, out_dim)
        )
        self.teacher_projector = nn.Sequential(
            nn.Linear(self.teacher.num_features, 2048),
            nn.GELU(),
            nn.Linear(2048, out_dim)
        )

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.register_buffer('center', torch.zeros(1, out_dim))
        self.momentum = momentum

    def forward(self, views, epoch=None, total_epochs=None, use_teacher=False):
        if use_teacher:
            return [self.teacher_projector(self.teacher(v)) for v in views[:2]]  # Only 2 global crops
        else:
            return [self.student_projector(self.student(v)) for v in views]

    def momentum_update(self):
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_s.data

    def update_center(self, teacher_outputs, epoch, total_epochs):
        batch_center = torch.cat(teacher_outputs).mean(dim=0, keepdim=True)

        # Fix: Use fixed momentum for first 10 epochs to stabilize early training
        if epoch < 10:
            center_momentum = self.momentum
        else:
            center_momentum = 1 - (1 - self.momentum) * (1 - epoch / total_epochs)

        self.center = self.center * center_momentum + batch_center * (1 - center_momentum)
