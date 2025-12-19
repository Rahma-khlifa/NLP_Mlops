import torch

print("PyTorch version :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())
print("Nombre de GPU   :", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Nom du GPU      :", torch.cuda.get_device_name(0))
    print("Version CUDA    :", torch.version.cuda)
else:
    print("⚠️ CUDA non disponible !")