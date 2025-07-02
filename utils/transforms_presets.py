
import torchvision.transforms as transforms


def generate_multicrop_views(image):
    global_crop = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    local_crop = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.2, 0.4)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),  # Resize to match ViT input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    views = [global_crop(image) for _ in range(2)]      # 2 global crops for student + teacher
    views += [local_crop(image) for _ in range(6)]      # 6 local crops for student only
    
    return views
presets = dict(
    transform_view1=dict(train=transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])),
    transform_view2=dict(train=transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
)
#         ),
#     AUGMENTATION = dict(
#     train=transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                      std=[0.229, 0.224, 0.225])
#         ])
#     ),

#     STRONG_AUGMENTATION = dict(
#     train=transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                      std=[0.229, 0.224, 0.225])
#     ]),
# ),

#     #  This one is for Question 4.
#     OXFORD_PETS=dict(
#         train=transforms.Compose([
#             transforms.Resize((256, 256)),  # Resize to a standard size

#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ]),
#         eval=transforms.Compose([
#             transforms.Resize((256, 256)),  # Resize to a standard size
#               # Resize to a standard size

#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         ])
#     ),
#     OXFORD_PETS_AUGMENTED = dict(
#     train=transforms.Compose([
#         transforms.Resize((256, 256)),  # Resize to a standard size

#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),  # ImageNet mean
#                              (0.229, 0.224, 0.225))  # ImageNet std
#     ]),
#     eval=transforms.Compose([
#         transforms.Resize((256, 256)),  # Resize to a standard size

#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                              (0.229, 0.224, 0.225))
#     ])
# )
# )