import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Net(Module):
    '''
    pytorch預測模型，包括LSTM時序預測層和Linear回歸輸出層
    可以根據自己的情況增加模型結構
    '''

    def __init__(self, config):
        super(Net, self).__init__()
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        self.linear = Linear(in_features=config.hidden_size, out_features=config.output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out, hidden


def train(config, logger, train_and_valid_data):
    if config.do_train_visualized:
        import visdom
        vis = visdom.Visdom(env='model_pytorch')

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()  # 先轉為Tensor
    train_loader = DataLoader(TensorDataset(train_X, train_Y),
                              batch_size=config.batch_size)  # DataLoader可自動生成可訓練的batch資料

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")  # CPU訓練還是GPU
    model = Net(config).to(device)  # 如果是GPU訓練， .to(device) 會把模型/資料複製到GPU顯存中
    if config.add_train:  # 如果是增量訓練，會先載入原模型參數
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()  # 這兩句是定義優化器和loss

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model.train()  # pytorch中，訓練時要轉換成訓練模式
        train_loss_array = []
        hidden_train = None
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device), _data[1].to(device)
            optimizer.zero_grad()  # 訓練前要將梯度資訊置 0
            pred_Y, hidden_train = model(_train_X, hidden_train)  # 這裡走的就是前向計算forward函數
            
            if not config.do_continue_train:
                hidden_train = None  # 如果非連續訓練，把hidden重置即可
            else:
                h_0, c_0 = hidden_train
                h_0.detach_(), c_0.detach_()  # 去掉梯度資訊
                hidden_train = (h_0, c_0)
            loss = criterion(pred_Y, _train_Y)  # 計算loss
            loss.backward()  # 將loss反向傳播
            optimizer.step()  # 用優化器更新參數
            train_loss_array.append(loss.item())
            global_step += 1
            if config.do_train_visualized and global_step % 100 == 0:  # 每一百步顯示一次
                vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
                         update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))

        # 以下為早停機制，當模型訓練連續config.patience個epoch都沒有使驗證集預測效果提升時，就停止，防止過擬合
        model.eval()  # pytorch中，預測時要轉換成預測模式
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y, hidden_valid = model(_valid_X, hidden_valid)
            if not config.do_continue_train: hidden_valid = None
            loss = criterion(pred_Y, _valid_Y)  # 驗證過程只有前向計算，無反向傳播過程
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
                    "The valid loss is {:.6f}.".format(valid_loss_cur))
        if config.do_train_visualized:  # 第一個train_loss_cur太大，導致沒有顯示在visdom中
            vis.line(X=np.array([epoch]), Y=np.array([train_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Train', opts=dict(showlegend=True))
            vis.line(X=np.array([epoch]), Y=np.array([valid_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Eval', opts=dict(showlegend=True))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.model_name)  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:  # 如果驗證集指標連續patience個epoch沒有提升，就停掉訓練
                logger.info(" The training stops early in epoch {}".format(epoch))
                break


def predict(config, test_X):
    # 獲取測試資料
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # 載入模型
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = Net(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))  # 載入模型參數

    # 先定義一個tensor保存預測結果
    result = torch.Tensor().to(device)

    # 預測過程
    model.eval()
    hidden_predict = None
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X, hidden_predict = model(data_X, hidden_predict)
        # if not config.do_continue_train: hidden_predict = None    # 實驗發現無論是否是連續訓練模式，把上一個time_step的hidden傳入下一個效果都更好
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()  # 先去梯度資訊，如果在gpu要轉到cpu，最後要返回numpy資料
