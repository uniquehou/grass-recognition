Config = {
    'output': (50, 50),
    'train_size': (227, 227),
    'train_batch_size': 30,
    'test_batch_size': 10,
    'lr': 0.001,    # 学习率
    'epochs': 100,    # 训练100轮
    'positive_weight': 1,
    'negative_weight': 3,
    'model_dir': 'trained_model',
    'best_model_dir': 'best_model',
    'pre_threhold': 128,
}