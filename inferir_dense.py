import os
import torch
import skimage.io
import torchvision.transforms as transforms
import torchxrayvision as xrv
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.cm as cm

# Script de inferência com geração de CAM (heatmap) para DenseNet
# Ajustado para usar os atributos corretos do wrapper DenseNet

def main():
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Carrega modelo DenseNet pretrained (res224)
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()

    # Dicionário para armazenar feature maps
    activation = {}

    # Hook na camada de features
    def get_activation(name):
        def hook(module, input, output):
            activation[name] = output.detach()
        return hook

    # Registra hook: usa model.features
    if hasattr(model, 'features'):
        model.features.register_forward_hook(get_activation('features'))
    else:
        raise AttributeError('DenseNet wrapper não possui atributo features')

    labels = model.pathologies

    # Pipeline de transformações: crop central e resize para 224x224
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(input_dir, fname)
        base = os.path.splitext(fname)[0]
        print(f"Processando {fname}...")

        # Salva cópia da imagem original
        orig_img = Image.open(path).convert("RGB")
        orig_img.save(os.path.join(output_dir, f"{base}_original.jpg"))
        orig_resized = orig_img.resize((224, 224))

        # Pré-processamento para inferência
        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255.0)  # converte 0-255 → -1024 a +1024

        # Garante único canal (C,H,W)
        if img.ndim == 3:
            img = img.mean(2)[None, ...]
        else:
            img = img[None, ...]

        # Aplica crop e resize
        img = transform(img)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device).float()

        # Inferência
        activation.clear()
        with torch.no_grad():
            outputs = model(img_tensor)[0].cpu().numpy()

        # Geração de CAM para classe de maior score
        # Obtém pesos da camada final classifier
        if hasattr(model, 'classifier'):
            weights = model.classifier.weight.detach().cpu().numpy()
        else:
            raise AttributeError('DenseNet wrapper não possui atributo classifier')

        idx = np.argmax(outputs)
        class_weights = weights[idx]
        fmap = activation['features'].squeeze(0).cpu().numpy()

        # Computa CAM
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for i, w in enumerate(class_weights):
            cam += w * fmap[i]
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        # Upsample CAM para 224x224 e converte em imagem
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        heatmap = cm.jet(np.array(cam_img) / 255.0)[..., :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap).convert('RGB')

        # Overlay (50% original + 50% heatmap)
        overlay = Image.blend(orig_resized, heatmap_img, alpha=0.5)
        overlay.save(os.path.join(output_dir, f"{base}_mapa.jpg"))

        # Salva CSV com scores
        df = pd.DataFrame({"label": labels, "score": outputs})
        csv_path = os.path.join(output_dir, f"{base}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Salvo: {base}.csv, {base}_original.jpg e {base}_mapa.jpg")

    # Gera list.txt com todos os .csv
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    with open(os.path.join(output_dir, 'list.txt'), 'w') as f:
        for cf in csv_files:
            f.write(f"{cf}\n")
    print('list.txt criado com sucesso')

if __name__ == '__main__':
    main()

