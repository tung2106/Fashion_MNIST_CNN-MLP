"""Smoke test: hai batch dữ liệu thật; không phải huấn luyện/đánh giá đầy đủ."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

root = Path(__file__).resolve().parents[2]
here = Path(__file__).resolve().parent
notebook = json.loads((here/'Fashion_MNIST_Hoc_Cham_Giai_Thich_Tung_Dong.ipynb').read_text(encoding='utf-8'))
codes = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
for index, code in enumerate(codes):
    compile(code, f'lesson_{index+1}', 'exec')
torch.set_num_threads(1)
source = datasets.FashionMNIST(root=root/'DU_LIEU', train=True, download=False, transform=transforms.ToTensor())
loader = DataLoader(Subset(source, range(128)), batch_size=64)
images, labels = next(iter(loader))
SEED, LR = 42, 0.001
exec(codes[3])  # Thiết bị và hàm seed.
exec(codes[12])  # Gradient ví dụ số.
for index in range(13, 23):
    exec(codes[index])  # MLP/CNN, một bước học, định nghĩa train/evaluate.
results = {}
for name, model_type in [('MLP', MLP), ('CNN', CNN)]:
    model = model_type().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    first_before = next(model.parameters()).detach().clone()
    train_loss, train_acc = train_one_epoch(model, loader, optimizer)
    assert not torch.equal(first_before, next(model.parameters()).detach())
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    val_loss, val_acc = evaluate(model, loader)
    assert all(torch.equal(state_before[key], value) for key, value in model.state_dict().items())
    assert np.isfinite([train_loss, train_acc, val_loss, val_acc]).all()
    assert 0 <= train_acc <= 1 and 0 <= val_acc <= 1
    assert model(images.to(DEVICE)).shape == (64, 10)
    results[name] = {'train_updates_weights': True, 'eval_preserves_weights': True, 'shape': [64,10]}
report = {'syntax_cells_pass':len(codes), 'runtime_smoke':results,
          'scope':'128 real train samples, unnormalized beyond ToTensor; technical check only, NOT full training or test evaluation',
          'not_run':'Full 5-epoch training, sklearn split, plots, pandas exports and Colab download are not executed by this local smoke test.'}
(here/'KIEM_TRA_NOTEBOOK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
