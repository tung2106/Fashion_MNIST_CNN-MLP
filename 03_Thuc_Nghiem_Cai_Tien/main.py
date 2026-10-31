"""Điểm bắt đầu: đọc cấu hình -> dữ liệu -> train -> tùy chọn test."""
import argparse
import json
import platform
import importlib.util
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import torch
import torchvision
from src.data import prepare_data, make_loader, load_test
from src.engine import fit
from src.evaluate import evaluate_models
from src.plots import plot_samples, plot_history
from src.utils import save_json

STAGE = Path(__file__).resolve().parent  # Không phụ thuộc terminal mở từ thư mục nào.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baseline.json', help='Tương đối với thư mục giai đoạn 3.')
    parser.add_argument('--inspect', action='store_true', help='Chỉ xem dữ liệu, không train.')
    parser.add_argument('--evaluate-test', action='store_true', help='Test sau khi train và chọn best bằng validation.')
    parser.add_argument('--data-dir', type=Path, default=STAGE.parent / 'DU_LIEU')
    args = parser.parse_args()
    config = json.loads((STAGE / args.config).read_text(encoding='utf-8'))
    if config['profile'] not in ('baseline', 'smoke'):
        parser.error('profile phải là baseline hoặc smoke')
    if config['epochs'] < 1 or config['batch_size'] < 1 or config['lr'] <= 0:
        parser.error('epochs, batch_size, lr phải dương')
    if not config['models'] or any(name not in ('MLP', 'CNN') for name in config['models']):
        parser.error('models phải chứa MLP và/hoặc CNN')
    if args.evaluate_test and (config['profile'] == 'smoke' or args.inspect):
        parser.error('Không test trong chế độ smoke/inspect.')
    # Giới hạn thread để smoke nhỏ không bị chậm vì CPU tạo quá nhiều luồng.
    torch.set_num_threads(2)
    if not args.inspect:
        try:
            on_colab = importlib.util.find_spec('google.colab') is not None
        except ModuleNotFoundError:
            on_colab = False
        if not on_colab or not torch.cuda.is_available():
            parser.error('Chỉ huấn luyện trên Google Colab có GPU. Không train trên máy cá nhân/CPU.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = STAGE / 'artifacts' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    for name in ['checkpoints', 'metrics', 'figures']:
        (run / name).mkdir()
    config.update(device=str(device), python=platform.python_version(), torch=str(torch.__version__),
                  torchvision=str(torchvision.__version__), evaluate_test=args.evaluate_test,
                  selection='min_validation_loss', data_dir=str(args.data_dir.resolve()))
    save_json(run / 'config.json', config)
    save_json(run / 'STATUS.json', {'stage': 'preparing'})
    print('Run:', run, flush=True)
    try:
        data = prepare_data(args.data_dir, config, run)
        images, labels = next(iter(make_loader(data['train'], config)))
        print('Train/validation:', len(data['train']), len(data['validation']))
        print('Batch images/labels:', tuple(images.shape), tuple(labels.shape))
        plot_samples(data, run)
        if args.inspect:
            save_json(run / 'STATUS.json', {'stage': 'data_inspected', 'trained': False, 'test_evaluated': False})
            return
        training_info = {}
        for name in config['models']:
            training_info[name] = fit(name, data, config, device, run)
            plot_history(name, run)
        if args.evaluate_test:
            test = load_test(args.data_dir, data)
            evaluate_models(make_loader(test, config), config, device, run, training_info)
        save_json(run / 'STATUS.json', {'stage': 'completed', 'profile': config['profile'],
                  'test_evaluated': args.evaluate_test, 'limits': ['one seed', 'no ablation', 'no exact resume']})
        print('COMPLETE:', run, flush=True)
    except Exception as error:
        save_json(run / 'STATUS.json', {'stage': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':  # Chạy khi gọi file; không tự chạy khi import.
    main()
