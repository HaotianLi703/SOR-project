# 如何修改参数
## 训练模型时
在train.py文件中，最下方主函数中修改 settings 字典即可  
其中：
- train_path为训练集路径
- dev_path为测试集路径
- batch_size为批量大小
- epoch_num为训练轮次
- ratio为训练数据占训练全集的比例，示例中0.00625表示使用了125个训练数据训练模型
- pre_model表示是否使用先前训练好的模型继续训练，如果使用，则输入模型路径，否则就输入None

模型保存的命名规则：model_训练数据量_batch大小_训练轮次

一个例子：
```python
if __name__ == "__main__":
    settings = {
        'hidden_size': 32,
        'type_size': 3,
        'train_path': 'data_retweet/train.pkl',
        'dev_path': 'data_retweet/dev.pkl',
        'batch_size': 10,
        'epoch_num': 250,
        'ratio': 0.00625,
        'current_date': datetime.date.today(),
        'pre_model' : 'model_125_10_250'
    }

    train(settings)
```
## 测试模型时  
同样，只需在test.py文件中更改主函数中的settings字典
- model_name表示测试时使用的模型路径
- test_path表示测试集路径

一个例子：
```python
if __name__ == "__main__":
    settings = {
        'model_name': 'model_125_10_270',
        'test_path':'data_retweet/test.pkl'
		'batch_size': 256
    }

    test(settings)
```
# 如何运行程序
可以直接在编辑器中运行train.py和test.py。train.py运行完成后会自动保存模型
