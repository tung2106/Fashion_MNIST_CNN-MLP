"""Vòng train/validation tách khỏi notebook; mọi biến cần thiết được truyền vào."""
import time
import pandas as pd
import torch
from torch import nn
from .models import MODEL_CLASSES
from .data import make_loader
from .utils import seed_all, save_json


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None  # Không truyền optimizer nghĩa là chỉ đo.
    model.train(training)  # True: train mode; False: eval mode.
    criterion = nn.CrossEntropyLoss()  # Nhận logits [B,10] và nhãn nguyên [B].
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)  # Xóa gradient, không xóa trọng số.
            logits = model(images)  # Forward: ảnh -> điểm của 10 lớp.
            loss = criterion(logits, labels)
            if training:
                loss.backward()  # Tính độ nhạy của loss theo từng tham số.
                optimizer.step()  # Đây mới là dòng sửa trọng số.
            n = labels.size(0)  # Batch cuối có thể ít ảnh hơn các batch trước.
            total_loss += loss.item() * n
            correct += (logits.argmax(1) == labels).sum().item()
            count += n
    if count == 0:
        raise ValueError('Loader rỗng, không thể tính loss/accuracy.')
    return {'loss': total_loss / count, 'accuracy': correct / count, 'samples': count}


def fit(name, data, config, device, run):
    seed_all(config['seed'])
    model = MODEL_CLASSES[name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loader = make_loader(data['train'], config, training=True)
    validation = make_loader(data['validation'], config)
    best_loss, best_epoch, history = float('inf'), 0, []
    started = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        train = run_epoch(model, loader, device, optimizer)
        val = run_epoch(model, validation, device)  # Không cập nhật trọng số.
        history.append(dict(epoch=epoch, train_loss=train['loss'], val_loss=val['loss'],
                            train_accuracy=train['accuracy'], val_accuracy=val['accuracy']))
        payload = dict(model_name=name, state_dict=model.state_dict(), epoch=epoch,
                       config=config, mean=data['mean'], std=data['std'], classes=data['classes'])
        if val['loss'] < best_loss:  # Hòa thì giữ epoch sớm hơn.
            best_loss, best_epoch = val['loss'], epoch
            torch.save(payload, run / 'checkpoints' / f'{name}_best.pt')
        # Last phục vụ kiểm tra trạng thái cuối, chưa hỗ trợ resume chính xác.
        torch.save({**payload, 'optimizer': optimizer.state_dict()},
                   run / 'checkpoints' / f'{name}_last.pt')
        pd.DataFrame(history).to_csv(run / 'metrics' / f'{name}_history.csv', index=False)
        save_json(run / 'STATUS.json', dict(stage='training', model=name, epoch=epoch))
        print(f"{name} {epoch}/{config['epochs']} train_loss={train['loss']:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f}", flush=True)
    info = dict(best_epoch=best_epoch, best_val_loss=best_loss,
                train_seconds=time.perf_counter() - started,
                parameters=sum(p.numel() for p in model.parameters()))
    save_json(run / 'metrics' / f'{name}_training.json', info)
    return info
